# ============================================================
# 08A_DIET_CLASSIFICATION.PY
# Purpose:
# Machine-learning counterpart to the Step 06 mixed models.
# Where 06 asked "does diet CHANGE hypothalamic LFP activity, on
# average, after controlling for covariates" (explanatory), this
# step asks the PREDICTIVE question:
#
#   RQ1: Can hypothalamic LFP activity be used to classify which
#        diet a recording belongs to (HF vs CTRL)?
#
# Design decisions (agreed with supervisor):
#   - Unit of observation is the RECORDING (all 125 clean
#     recordings in the analysis window, Feb 19 - Apr 2; baseline +
#     diet + recovery). No aggregation to mouse level.
#   - Recordings from the same mouse are NOT independent (same
#     brain / electrode / animal). This is the exact non-
#     independence the 06 mixed model handled with a random
#     intercept (1|mouse). The ML analogue is a mouse-grouped
#     split: Leave-One-Mouse-Out (LOMO) cross-validation, so no
#     mouse ever appears in both train and test (prevents the model
#     from learning mouse identity instead of diet = data leakage).
#   - Features are LFP-derived ONLY. Body weight is deliberately
#     EXCLUDED: HF mice weigh more by construction, so body weight
#     trivially predicts diet; including it would answer the wrong
#     question. This mirrors Step 06, where body weight had to be
#     controlled because it masks the diet signal in raw data.
#
# What is reported:
#   - Primary metric: balanced accuracy + ROC-AUC, from pooled
#     out-of-fold predictions under LOMO (each recording is scored
#     only while its mouse is held out).
#   - A naive random-split (StratifiedKFold) result is reported
#     ALONGSIDE, purely as a leakage demonstration: if random-split
#     accuracy is much higher than LOMO, that gap IS the leakage.
#     (Same logic as 06's Mann-Whitney vs mixed-model contrast.)
#   - A permutation test (mouse-level label shuffle) gives the
#     chance level and a p-value for the LOMO balanced accuracy.
#   - Two models: L2-regularised logistic regression (primary,
#     interpretable, robust at small n) and random forest
#     (non-linear comparison).
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05c_oscillation_episodes_<CABLE>/
#       05c_features_combined_with_episodes_<CABLE>.csv
#
# Output:
#   outputs/08a_diet_classification_<CABLE>/
#       08a_diet_classification_results_<CABLE>.csv   (long format)
#       08a_diet_classification_summary_<CABLE>.txt   (readable)
#       08a_diet_classification_<CABLE>.png           (4-panel figure)
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
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             accuracy_score, confusion_matrix, roc_curve)

# statsmodels-style: keep the terminal readable. Convergence and
# chance level are reported explicitly below.
warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"
RANDOM_STATE = 0
N_PERMUTATIONS = 1000          # mouse-level label shuffles
N_SPLITS_RANDOM = 5            # folds for the naive random-split baseline

# Resolve project root from this script's location (scripts/ -> root),
# so paths work regardless of the project folder name.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}", f"05a_band_powers_{CABLE}.csv"
)
COMBINED_PATH = os.path.join(
    OUTPUT_DIR, f"05c_oscillation_episodes_{CABLE}",
    f"05c_features_combined_with_episodes_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"08a_diet_classification_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"08a_diet_classification_results_{CABLE}.csv")
OUT_TXT = os.path.join(OUT_DIR, f"08a_diet_classification_summary_{CABLE}.txt")
OUT_PNG = os.path.join(OUT_DIR, f"08a_diet_classification_{CABLE}.png")

# Positive class = HF (so ROC-AUC reads "how well HF is detected")
POSITIVE_GROUP = "HF"

# Bands (must match 05a column scheme)
BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]

# --- Feature set: LFP-derived only (NO body weight, estrous, phase) ---
# 6 log band powers + 6 relative band powers
LOG_POWER_FEATS = [f"log_{b}_abs" for b in BANDS]     # created below
REL_POWER_FEATS = [f"{b}_rel" for b in BANDS]
# 5 log band ratios (already in the 05c combined table)
LOG_RATIO_FEATS = [
    "log_theta_delta", "log_low_gamma_theta", "log_high_gamma_theta",
    "log_fast_gamma_theta", "log_beta_theta",
]
# 12 oscillation-episode features (3 bands x 4 metrics), matching the
# 06 exploratory set (episode_rate, mean_duration_sec, mean_amplitude,
# fraction_of_time); n_episodes / threshold are excluded (redundant /
# detection parameter).
EPISODE_BANDS = ["beta", "low_gamma", "high_gamma"]
EPISODE_METRICS = ["episode_rate", "mean_duration_sec",
                   "mean_amplitude", "fraction_of_time"]
EPISODE_FEATS = [f"{b}_{m}" for b in EPISODE_BANDS for m in EPISODE_METRICS]

ALL_FEATURES = LOG_POWER_FEATS + REL_POWER_FEATS + LOG_RATIO_FEATS + EPISODE_FEATS

# Colours (consistent with 06 figures)
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
# ------------------------------------------------------------
# 05a (band powers + metadata) is the base; the 05c combined table
# (log ratios + episode features) is merged on differential_file
# (1:1, same clean set). log_<band>_abs columns are added here as a
# statistical transform, exactly as in 06.
# ============================================================

for p, step in [(BAND_POWERS_PATH, "05a"), (COMBINED_PATH, "05c")]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {step} output:\n{p}")

df_05a = pd.read_csv(BAND_POWERS_PATH)
df_05c = pd.read_csv(COMBINED_PATH)

# Log band powers (statistical transform)
for b in BANDS:
    df_05a[f"log_{b}_abs"] = np.log10(df_05a[f"{b}_abs"])

merge_cols = ["differential_file"] + LOG_RATIO_FEATS + EPISODE_FEATS
df = df_05a.merge(df_05c[merge_cols], on="differential_file",
                  how="left", validate="one_to_one")

print(f"\nLoaded feature table: {df.shape[0]} recordings x "
      f"{df.shape[1]} columns")

# Sanity: features present + complete
missing = [c for c in ALL_FEATURES if c not in df.columns]
if missing:
    raise KeyError(f"Missing feature columns: {missing}")
n_nan = df[ALL_FEATURES].isna().any(axis=1).sum()
if n_nan > 0:
    raise ValueError(f"{n_nan} recording(s) have NaN in a feature — "
                     f"check 05a/05c outputs.")


# ============================================================
# 3. TARGET, FEATURES, GROUPS
# ============================================================

X = df[ALL_FEATURES].to_numpy(dtype=float)
y = (df["group"] == POSITIVE_GROUP).astype(int).to_numpy()   # HF = 1
groups = df["mouse"].to_numpy()

n_obs = len(y)
n_mice = len(np.unique(groups))
n_hf = int(y.sum())
n_ctrl = int((1 - y).sum())

# Majority-class baseline accuracy (context for "raw" accuracy)
majority_acc = max(n_hf, n_ctrl) / n_obs

print(f"  observations : {n_obs} recordings from {n_mice} mice")
print(f"  class balance: HF={n_hf}  CTRL={n_ctrl} "
      f"(majority-class accuracy = {majority_acc:.3f})")
print(f"  features     : {len(ALL_FEATURES)} (LFP only; body weight "
      f"excluded by design)")


# ============================================================
# 4. MODELS (as sklearn Pipelines so scaling is fit INSIDE each
#    training fold — never on the held-out data)
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
        ("scale", StandardScaler()),   # harmless for RF; keeps interface uniform
        ("clf", RandomForestClassifier(n_estimators=500,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE)),
    ])


MODELS = {"logistic_regression": make_logreg,
          "svm_rbf": make_svm,
          "random_forest": make_rf}


# ============================================================
# 5. EVALUATION HELPERS
# ------------------------------------------------------------
# For each CV scheme we collect POOLED out-of-fold predictions
# (one prediction per recording, produced while its fold is held
# out) and compute metrics on the pool. Under LOMO a single fold
# contains one mouse = one class, so per-fold AUC is undefined;
# pooling across folds is the correct way to score.
# ============================================================

def pooled_predictions(make_model, X, y, cv, groups=None):
    """Return pooled out-of-fold (proba_HF, predicted_label)."""
    model = make_model()
    proba = cross_val_predict(model, X, y, cv=cv, groups=groups,
                              method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return proba, pred


def score(y_true, proba, pred):
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
    }


logo = LeaveOneGroupOut()
skf = StratifiedKFold(n_splits=N_SPLITS_RANDOM, shuffle=True,
                      random_state=RANDOM_STATE)


# ============================================================
# 6. RUN BOTH MODELS UNDER BOTH CV SCHEMES
# ============================================================

print("\n" + "=" * 60)
print("CROSS-VALIDATED CLASSIFICATION")
print("=" * 60)

rows = []
lomo_store = {}   # keep LOMO proba for the figure + permutation

for model_name, make_model in MODELS.items():
    # --- LOMO (honest, mouse-grouped) ---
    proba_l, pred_l = pooled_predictions(make_model, X, y, logo, groups)
    m_l = score(y, proba_l, pred_l)
    lomo_store[model_name] = proba_l
    for metric, val in m_l.items():
        rows.append({"model": model_name, "cv_scheme": "leave_one_mouse_out",
                     "metric": metric, "value": val})
    print(f"\n{model_name} — LEAVE-ONE-MOUSE-OUT (primary):")
    print(f"  balanced_accuracy = {m_l['balanced_accuracy']:.3f}")
    print(f"  roc_auc           = {m_l['roc_auc']:.3f}")
    print(f"  accuracy          = {m_l['accuracy']:.3f} "
          f"(majority baseline {majority_acc:.3f})")

    # --- Random split (naive; leakage demonstration only) ---
    proba_r, pred_r = pooled_predictions(make_model, X, y, skf, None)
    m_r = score(y, proba_r, pred_r)
    for metric, val in m_r.items():
        rows.append({"model": model_name, "cv_scheme": "random_kfold",
                     "metric": metric, "value": val})
    print(f"{model_name} — RANDOM {N_SPLITS_RANDOM}-FOLD "
          f"(leakage demo, NOT valid):")
    print(f"  balanced_accuracy = {m_r['balanced_accuracy']:.3f}  "
          f"(gap vs LOMO = "
          f"{m_r['balanced_accuracy'] - m_l['balanced_accuracy']:+.3f})")

results_df = pd.DataFrame(rows)


# ============================================================
# 7. PERMUTATION TEST (mouse-level label shuffle)
# ------------------------------------------------------------
# The label is a MOUSE property, so a valid null shuffles the
# group label across MICE (keeping each mouse's recordings
# together), then re-runs the full LOMO pipeline. Shuffling at the
# recording level would leak the true structure and give an easy,
# invalid null. Done for the primary model (logistic regression).
# p = (1 + #{null >= observed}) / (1 + N_PERMUTATIONS).
# ============================================================

print("\n" + "=" * 60)
print(f"PERMUTATION TEST  (logistic regression, LOMO, "
      f"{N_PERMUTATIONS} shuffles)")
print("=" * 60)

# One label per mouse (mouse -> HF/CTRL)
mouse_series = df.groupby("mouse")["group"].first()
mouse_ids = mouse_series.index.to_numpy()
mouse_labels = (mouse_series == POSITIVE_GROUP).astype(int).to_numpy()

observed_bacc = results_df.query(
    "model == 'logistic_regression' and "
    "cv_scheme == 'leave_one_mouse_out' and metric == 'balanced_accuracy'"
)["value"].iloc[0]

rng = np.random.default_rng(RANDOM_STATE)
null_bacc = np.empty(N_PERMUTATIONS)
for i in range(N_PERMUTATIONS):
    permuted = rng.permutation(mouse_labels)
    mapping = dict(zip(mouse_ids, permuted))
    y_perm = np.array([mapping[g] for g in groups])
    _, pred_p = pooled_predictions(make_logreg, X, y_perm, logo, groups)
    null_bacc[i] = balanced_accuracy_score(y_perm, pred_p)
    if (i + 1) % 200 == 0:
        print(f"  ...{i + 1}/{N_PERMUTATIONS} permutations done")

perm_p = (1 + np.sum(null_bacc >= observed_bacc)) / (1 + N_PERMUTATIONS)
null_mean = float(null_bacc.mean())

print(f"\n  observed balanced accuracy = {observed_bacc:.3f}")
print(f"  permutation-null mean      = {null_mean:.3f}  "
      f"(empirical chance level)")
print(f"  permutation p-value        = {perm_p:.4f}")

rows.append({"model": "logistic_regression", "cv_scheme": "leave_one_mouse_out",
             "metric": "perm_null_mean_balanced_accuracy", "value": null_mean})
rows.append({"model": "logistic_regression", "cv_scheme": "leave_one_mouse_out",
             "metric": "perm_p_value_balanced_accuracy", "value": perm_p})
results_df = pd.DataFrame(rows)


# ============================================================
# 8. SAVE — LONG-FORMAT CSV
# ============================================================

results_df = results_df[["model", "cv_scheme", "metric", "value"]]
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved results CSV:\n{OUT_CSV}")


# ============================================================
# 9. SAVE — HUMAN-READABLE TXT
# ============================================================

def get(model, cv, metric):
    q = results_df.query("model == @model and cv_scheme == @cv "
                         "and metric == @metric")
    return q["value"].iloc[0] if len(q) else float("nan")

lines = []
lines.append("=" * 70)
lines.append(f"08A DIET CLASSIFICATION (HF vs CTRL) — {CABLE}")
lines.append("=" * 70)
lines.append("")
lines.append(f"Observations : {n_obs} recordings / {n_mice} mice "
             f"(HF {n_hf} / CTRL {n_ctrl})")
lines.append(f"Features     : {len(ALL_FEATURES)} LFP-derived "
             f"(6 log power, 6 rel power, 5 log ratio, 12 episode)")
lines.append(f"               body weight EXCLUDED by design")
lines.append(f"Primary CV   : leave-one-mouse-out (no mouse in train+test)")
lines.append(f"Majority-class accuracy (context) : {majority_acc:.3f}")
lines.append("")
lines.append("-" * 70)
lines.append(f"{'model':<22s} {'CV scheme':<20s} {'bal_acc':>8s}"
             f" {'roc_auc':>8s} {'acc':>7s}")
lines.append("-" * 70)
for model in MODELS:
    for cv, label in [("leave_one_mouse_out", "LOMO (valid)"),
                      ("random_kfold", "random (leakage)")]:
        lines.append(f"{model:<22s} {label:<20s} "
                     f"{get(model, cv, 'balanced_accuracy'):>8.3f}"
                     f" {get(model, cv, 'roc_auc'):>8.3f}"
                     f" {get(model, cv, 'accuracy'):>7.3f}")
lines.append("")
lines.append("-" * 70)
lines.append("PERMUTATION TEST (logistic regression, LOMO)")
lines.append("-" * 70)
lines.append(f"  observed balanced accuracy : {observed_bacc:.3f}")
lines.append(f"  empirical chance level     : {null_mean:.3f}")
lines.append(f"  p-value ({N_PERMUTATIONS} shuffles)     : {perm_p:.4f}")
lines.append("")
lines.append("Reading guide:")
lines.append("  - LOMO is the valid estimate. The random-split row is shown")
lines.append("    only to expose leakage: a large random-minus-LOMO gap means")
lines.append("    the classifier was reading mouse identity, not diet.")
lines.append("  - Body weight is excluded on purpose; this tests LFP itself,")
lines.append("    complementing the 06 mixed model (which controlled for it).")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary:\n{OUT_TXT}")


# ============================================================
# 10. FIGURE (4 panels)
#   A) LOMO vs random balanced accuracy, both models (leakage view)
#   B) ROC curve, LOMO, both models
#   C) confusion matrix, logistic regression, LOMO
#   D) permutation-null histogram with observed value + chance
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# --- A: LOMO vs random ---
ax = axes[0, 0]
model_list = list(MODELS.keys())
x = np.arange(len(model_list))
w = 0.38
lomo_vals = [get(m, "leave_one_mouse_out", "balanced_accuracy")
             for m in model_list]
rand_vals = [get(m, "random_kfold", "balanced_accuracy")
             for m in model_list]
ax.bar(x - w / 2, lomo_vals, w, color=C_ACCENT, alpha=0.9,
       label="LOMO (valid)")
ax.bar(x + w / 2, rand_vals, w, color="#B0B0B0", alpha=0.9,
       label="random k-fold (leakage)")
ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="chance (0.5)")
ax.set_xticks(x)
ax.set_xticklabels([m.replace("_", "\n") for m in model_list], fontsize=9)
ax.set_ylabel("Balanced accuracy")
ax.set_ylim(0, 1)
ax.set_title("A. Valid vs leaky CV", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=8)

# --- B: ROC (LOMO) ---
ax = axes[0, 1]
roc_colors = {"logistic_regression": C_HF, "svm_rbf": "#8172B3",
              "random_forest": C_CTRL}
for model_name, col in roc_colors.items():
    proba = lomo_store[model_name]
    fpr, tpr, _ = roc_curve(y, proba)
    auc = get(model_name, "leave_one_mouse_out", "roc_auc")
    ax.plot(fpr, tpr, color=col, lw=1.8,
            label=f"{model_name.replace('_', ' ')} (AUC={auc:.2f})")
ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("B. ROC (leave-one-mouse-out)", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")

# --- C: confusion matrix (logistic, LOMO) ---
ax = axes[1, 0]
pred_lr = (lomo_store["logistic_regression"] >= 0.5).astype(int)
cm = confusion_matrix(y, pred_lr, labels=[0, 1])
# vmin=0 anchors the colour scale at zero so the smallest cell is not
# forced to near-white; otherwise the min cell renders pale while its
# count still gets white text (min > max/2 here), making it unreadable.
im = ax.imshow(cm, cmap="Blues", vmin=0)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["CTRL", "HF"]); ax.set_yticklabels(["CTRL", "HF"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("C. Confusion matrix\n(logistic, LOMO)",
             fontweight="bold", loc="left")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=13, fontweight="bold")

# --- D: permutation null ---
ax = axes[1, 1]
ax.hist(null_bacc, bins=30, color="#B0B0B0", alpha=0.85,
        label="null (shuffled labels)")
ax.axvline(null_mean, color="grey", ls="--", lw=1.2,
           label=f"chance ({null_mean:.2f})")
ax.axvline(observed_bacc, color=C_HF, lw=2.0,
           label=f"observed ({observed_bacc:.2f})")
ax.set_xlabel("Balanced accuracy")
ax.set_ylabel("Permutations")
ax.set_title(f"D. Permutation null (p={perm_p:.3f})",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=8)

fig.suptitle(f"Step 08a — diet classification from LFP ({CABLE}, "
             f"{n_obs} recordings / {n_mice} mice)",
             fontsize=13, fontweight="bold", y=1.00)
fig.text(0.5, 0.005,
         "LFP-only features, body weight excluded by design. LOMO is the "
         "valid estimate; the random-split bar exposes mouse-identity leakage.",
         ha="center", fontsize=9, color="#444444", style="italic")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved figure:\n{OUT_PNG}")

print("\nSTEP 08A (diet classification) finished successfully.")