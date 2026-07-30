# ============================================================
# 08C_BAND_INFORMATIVENESS.PY
# Purpose:
#   RQ3: Which frequency band is the most informative for
#        predicting nutritional state (HF vs CTRL diet), and which
#        gives the best vs the worst prediction accuracy?
#
# Approach:
#   Train the SAME classifier separately on each band's features
#   and compare cross-validated performance band by band. The band
#   with the highest score is the most informative; the lowest is
#   the least. This answers RQ3 directly, unlike reading feature
#   importances from one big model (which is harder to interpret
#   and mixes bands together).
#
#   Per-band feature set (uniform across all six bands):
#     [ log_<band>_abs , <band>_rel ]   -> 2 features per band
#   i.e. the band's absolute power (log) and its relative power
#   (share of the 1-140 Hz spectrum). Ratios and episode features
#   are NOT band-exclusive (ratios span two bands; episodes exist
#   only for beta/low_gamma/high_gamma), so they are left out here
#   to keep every band on an equal, comparable footing. An
#   "all_bands_power" set (all 12 power features) is included as a
#   reference ceiling.
#
# Context from 08a: LFP as a whole does not classify diet above
#   chance, so per-band scores are expected to sit around chance
#   too; RQ3 is then a RELATIVE ranking ("least uninformative"),
#   and we check with a permutation test whether any single band
#   rises above chance on its own.
#
# Validation / metrics (as in 08a):
#   - Leave-One-Mouse-Out, pooled out-of-fold predictions
#   - balanced accuracy, ROC-AUC, accuracy
#   - three models: logistic regression, SVM (RBF), random forest
#   - mouse-level label-shuffle permutation on the primary model
#     (logistic regression), per band.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05c_oscillation_episodes_<CABLE>/
#       05c_features_combined_with_episodes_<CABLE>.csv
#
# Output:
#   outputs/08c_band_informativeness_<CABLE>/
#       08c_band_informativeness_results_<CABLE>.csv   (long format)
#       08c_band_informativeness_summary_<CABLE>.txt   (readable, ranked)
#       08c_band_informativeness_<CABLE>.png           (figure)
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
                             accuracy_score)

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

OUT_DIR = os.path.join(OUTPUT_DIR, f"08c_band_informativeness_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"08c_band_informativeness_results_{CABLE}.csv")
OUT_TXT = os.path.join(OUT_DIR, f"08c_band_informativeness_summary_{CABLE}.txt")
OUT_PNG = os.path.join(OUT_DIR, f"08c_band_informativeness_{CABLE}.png")

# Bands in ascending frequency order (for a readable x-axis)
BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]

# Per-band feature set: absolute (log) + relative power
def band_features(band):
    return [f"log_{band}_abs", f"{band}_rel"]

# Feature sets: one per band + an all-bands reference
FEATURE_SETS = {b: band_features(b) for b in BANDS}
FEATURE_SETS["all_bands_power"] = [f"log_{b}_abs" for b in BANDS] + \
                                  [f"{b}_rel" for b in BANDS]

MODEL_NAMES = ["logistic_regression", "svm_rbf", "random_forest"]
PRIMARY_MODEL = "logistic_regression"

BAND_COLOR = "#4C72B0"
REF_COLOR = "#937860"

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
df_05c = pd.read_csv(COMBINED_PATH)   # not strictly needed, kept for parity

for b in BANDS:
    df_05a[f"log_{b}_abs"] = np.log10(df_05a[f"{b}_abs"])

df = df_05a
print(f"\nLoaded feature table: {df.shape[0]} recordings x {df.shape[1]} cols")

all_feats = sorted({c for cols in FEATURE_SETS.values() for c in cols})
missing = [c for c in all_feats if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}")
if df[all_feats].isna().any(axis=1).sum() > 0:
    raise ValueError("NaN in features — check 05a outputs.")


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
# 4. MODELS
# ============================================================

def make_model(name):
    if name == "logistic_regression":
        clf = LogisticRegression(C=1.0, max_iter=5000,
                                 class_weight="balanced",
                                 random_state=RANDOM_STATE)
    elif name == "svm_rbf":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                  class_weight="balanced", random_state=RANDOM_STATE)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=500,
                                     class_weight="balanced",
                                     random_state=RANDOM_STATE)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


logo = LeaveOneGroupOut()


def pooled_predictions(pipe, X, y, groups):
    proba = cross_val_predict(pipe, X, y, cv=logo, groups=groups,
                              method="predict_proba")[:, 1]
    return proba, (proba >= 0.5).astype(int)


def score(y_true, proba, pred):
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
    }


# ============================================================
# 5. RUN EACH BAND (+ reference) x EACH MODEL
# ============================================================

print("\n" + "=" * 64)
print("PER-BAND DIET CLASSIFICATION (LOMO)")
print("=" * 64)

rows = []
for fs_name, cols in FEATURE_SETS.items():
    X = df[cols].to_numpy(dtype=float)
    print(f"\n[{fs_name}]  ({len(cols)} features)")
    for model_name in MODEL_NAMES:
        proba, pred = pooled_predictions(make_model(model_name), X, y, groups)
        m = score(y, proba, pred)
        for metric, val in m.items():
            rows.append({"feature_set": fs_name, "model": model_name,
                         "metric": metric, "value": val})
        print(f"  {model_name:<20s} bal_acc={m['balanced_accuracy']:.3f}"
              f"  auc={m['roc_auc']:.3f}  acc={m['accuracy']:.3f}")

results_df = pd.DataFrame(rows)


# ============================================================
# 6. PERMUTATION TEST (logistic, per band; mouse-level shuffle)
# ============================================================

print("\n" + "=" * 64)
print(f"PERMUTATION TEST (logistic regression, LOMO, "
      f"{N_PERMUTATIONS} shuffles per band)")
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
        _, pred_p = pooled_predictions(make_model("logistic_regression"),
                                       X, y_perm, groups)
        null[i] = balanced_accuracy_score(y_perm, pred_p)

    p_val = (1 + np.sum(null >= observed)) / (1 + N_PERMUTATIONS)
    perm_summary[fs_name] = {"observed": observed,
                             "null_mean": float(null.mean()),
                             "p_value": p_val}
    print(f"  {fs_name:<16s} observed={observed:.3f}  "
          f"chance={null.mean():.3f}  p={p_val:.4f}")

    rows.append({"feature_set": fs_name, "model": "logistic_regression",
                 "metric": "perm_p_value_balanced_accuracy", "value": p_val})

results_df = pd.DataFrame(rows)


# ============================================================
# 7. RANKING (primary model = logistic; by ROC-AUC)
# ============================================================

rank = (results_df.query("model == @PRIMARY_MODEL and metric == 'roc_auc' "
                         "and feature_set in @BANDS")
        .sort_values("value", ascending=False)
        .reset_index(drop=True))
most_informative = rank.iloc[0]["feature_set"]
least_informative = rank.iloc[-1]["feature_set"]


# ============================================================
# 8. SAVE — LONG-FORMAT CSV
# ============================================================

results_df = results_df[["feature_set", "model", "metric", "value"]]
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved results CSV:\n{OUT_CSV}")


# ============================================================
# 9. SAVE — HUMAN-READABLE TXT (ranked)
# ============================================================

def get(fs, model, metric):
    q = results_df.query("feature_set == @fs and model == @model "
                         "and metric == @metric")
    return q["value"].iloc[0] if len(q) else float("nan")

lines = []
lines.append("=" * 72)
lines.append(f"08C BAND INFORMATIVENESS FOR DIET — {CABLE}")
lines.append("=" * 72)
lines.append("")
lines.append(f"Observations : {n_obs} recordings / {n_mice} mice "
             f"(HF {n_hf} / CTRL {n_ctrl})")
lines.append(f"Per-band features : log(abs power) + relative power (2 each)")
lines.append(f"Validation   : leave-one-mouse-out")
lines.append(f"Chance / majority-class accuracy : 0.500 / {majority_acc:.3f}")
lines.append("")
lines.append("-" * 72)
lines.append(f"{'band':<16s} {'model':<22s} {'bal_acc':>8s}"
             f" {'roc_auc':>8s} {'acc':>7s}")
lines.append("-" * 72)
for fs in list(BANDS) + ["all_bands_power"]:
    for model in MODEL_NAMES:
        lines.append(f"{fs:<16s} {model:<22s} "
                     f"{get(fs, model, 'balanced_accuracy'):>8.3f}"
                     f" {get(fs, model, 'roc_auc'):>8.3f}"
                     f" {get(fs, model, 'accuracy'):>7.3f}")
    lines.append("")
lines.append("-" * 72)
lines.append(f"RANKING by ROC-AUC ({PRIMARY_MODEL}); + permutation p")
lines.append("-" * 72)
lines.append(f"{'rank':<5s} {'band':<14s} {'roc_auc':>8s} "
             f"{'bal_acc':>8s} {'perm_p':>8s}")
for i, r in rank.iterrows():
    fs = r["feature_set"]
    lines.append(f"{i + 1:<5d} {fs:<14s} {r['value']:>8.3f} "
                 f"{get(fs, PRIMARY_MODEL, 'balanced_accuracy'):>8.3f} "
                 f"{perm_summary[fs]['p_value']:>8.4f}")
lines.append("")
lines.append(f"Most informative band : {most_informative}")
lines.append(f"Least informative band: {least_informative}")
lines.append("")
lines.append("Reading guide:")
lines.append("  - Ranking is RELATIVE. If no band's permutation p < 0.05, no")
lines.append("    single band classifies diet above chance on its own; the")
lines.append("    ranking then says which band is least uninformative, not")
lines.append("    that it is a reliable predictor.")
lines.append("  - Compare with Step 06: the band that was significant there")
lines.append("    (delta) is worth checking against its rank here.")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary:\n{OUT_TXT}")


# ============================================================
# 10. FIGURE (2 panels)
#   A) per-band balanced accuracy, three models (+ reference)
#   B) per-band ROC-AUC (logistic), ranked, with chance line and
#      permutation significance marks
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
plot_sets = list(BANDS) + ["all_bands_power"]

# --- A: balanced accuracy, models grouped ---
ax = axes[0]
x = np.arange(len(plot_sets))
w = 0.26
model_colors = {"logistic_regression": "#C44E52",
                "svm_rbf": "#8172B3",
                "random_forest": "#4C72B0"}
for j, model in enumerate(MODEL_NAMES):
    vals = [get(fs, model, "balanced_accuracy") for fs in plot_sets]
    ax.bar(x + (j - 1) * w, vals, w, color=model_colors[model],
           alpha=0.9, label=model.replace("_", " "))
ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="chance (0.5)")
ax.set_xticks(x)
ax.set_xticklabels([s.replace("_", "\n") for s in plot_sets],
                   rotation=0, fontsize=8)
ax.set_ylabel("Balanced accuracy")
ax.set_ylim(0, 1)
ax.set_title("A. Per-band diet classification (LOMO)",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=8, ncol=2)

# --- B: ranked ROC-AUC (logistic), permutation stars ---
ax = axes[1]
rank_bands = rank["feature_set"].tolist()
rank_auc = rank["value"].tolist()
colors = [BAND_COLOR] * len(rank_bands)
bars = ax.barh(range(len(rank_bands))[::-1], rank_auc, color=colors,
               alpha=0.9)
ax.axvline(0.5, color="grey", ls="--", lw=0.8, label="chance (0.5)")
ax.set_yticks(range(len(rank_bands))[::-1])
ax.set_yticklabels(rank_bands)
ax.set_xlabel("ROC-AUC (logistic, LOMO)")
ax.set_xlim(0, 1)
ax.set_title("B. Bands ranked by AUC\n(* = permutation p < 0.05)",
             fontweight="bold", loc="left")
for k, fs in enumerate(rank_bands):
    ypos = (len(rank_bands) - 1) - k
    star = " *" if perm_summary[fs]["p_value"] < 0.05 else ""
    ax.text(rank_auc[k] + 0.01, ypos, f"{rank_auc[k]:.2f}{star}",
            va="center", fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.suptitle(f"Step 08c — which band predicts diet ({CABLE}, "
             f"{n_obs} recordings / {n_mice} mice)",
             fontsize=13, fontweight="bold", y=1.02)
fig.text(0.5, -0.02,
         f"Most informative: {most_informative}; least: {least_informative}. "
         f"Ranking is relative — check the permutation p-values for whether "
         f"any band beats chance.",
         ha="center", fontsize=9, color="#444444", style="italic")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved figure:\n{OUT_PNG}")

print(f"\nMost informative band : {most_informative}")
print(f"Least informative band: {least_informative}")
print("\nSTEP 08C (band informativeness) finished successfully.")