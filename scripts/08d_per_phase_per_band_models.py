# ============================================================
# 08D_PER_PHASE_PER_BAND_MODELS.PY
# Purpose:
#   RQ1 + RQ2 revisited following PROFESSOR's supervision meeting
#   (Jul 9 2026, "Universitätsstraße 14" meeting).
#
#   PROFESSOR's directives (transcribed and cleaned up):
#     1. Do NOT use body weight as an LFP feature — weight is a
#        strong correlate of group (HF vs CTRL) by design, so
#        including it inflates classification accuracy through the
#        label, not through the physiology. Weight is a SEPARATE
#        prediction TARGET.
#     2. Build separate models per estrous cycle phase (A, B, C, D)
#        instead of pooling all phases with dummy-coded phase as a
#        feature. Rationale: the LFP signature of diet may only
#        emerge inside a specific phase; pooling washes it out.
#     3. Build separate models per frequency band. Six bands x four
#        phases = 24 models per target per cable. Rationale: any
#        one-model-that-uses-everything answer would not say WHERE
#        the predictive information lives.
#     4. Drop band ratios. Relative band power is already a ratio
#        (band / total power), so band ratios add nothing new.
#     5. Keep the two cables separate (Cable 1 vs. Cable 3), as in
#        every other script — no pooling across cables.
#     6. Models to use:
#          group    (classification)  ->  SVM-RBF + Random Forest
#          weight   (regression)      ->  Random Forest
#        (PROFESSOR named these two families explicitly.)
#
# Per-band feature block:
#   For a given band b, features = [log_<b>_abs, <b>_rel].
#     log_<b>_abs : log10 absolute band power (magnitude, log scale)
#     <b>_rel    : relative band power (band / total) — the
#                  "normalised" form PROFESSOR emphasised.
#   Two features per model keep the models informative without
#   reintroducing ratios or cross-band contamination.
#
# Validation / null:
#   - Leave-One-Mouse-Out CV (grouped by mouse), same as 08a/08b —
#     no mouse in train and test at the same time.
#   - Pooled out-of-fold predictions.
#   - Permutation test on the primary model of each task, with the
#     correct level of shuffling:
#         group  -> shuffle at MOUSE level (fixed per-mouse label)
#         weight -> shuffle at RECORDING level (varies within mouse)
#
# Metrics:
#   group     : balanced accuracy, ROC-AUC, accuracy
#   weight    : R^2, MAE (grams)
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#     (05c oscillation-episode features are NOT used here — this
#      script is deliberately about band power alone, per PROFESSOR.)
#
# Output:
#   outputs/08d_per_phase_per_band_<CABLE>/
#       08d_results_long_<CABLE>.csv       (long-format results)
#       08d_summary_<CABLE>.txt            (readable summary)
#       08d_heatmap_group_<CABLE>.png      (phase x band, bal_acc)
#       08d_heatmap_weight_<CABLE>.png     (phase x band, R^2)
#
# Usage:
#   Set CABLE = "Cable1" or "Cable3" in SETTINGS, run twice.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             accuracy_score, r2_score, mean_absolute_error)

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"                 # switch to "Cable3" for the parallel run
RANDOM_STATE = 0
N_PERMUTATIONS = 1000
POSITIVE_GROUP = "HF"

# A recording enters a per-phase model only if that phase has
# enough data to be meaningful. PROFESSOR did not name a threshold;
# these are conservative defaults — a phase must have at least
# MIN_RECORDINGS observations spread across at least MIN_MICE mice
# AND both diet groups must be present. Under these are set nan.
MIN_RECORDINGS_PER_PHASE = 12
MIN_MICE_PER_PHASE = 6

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}", f"05a_band_powers_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"08d_per_phase_per_band_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"08d_results_long_{CABLE}.csv")
OUT_TXT = os.path.join(OUT_DIR, f"08d_summary_{CABLE}.txt")
OUT_PNG_GROUP = os.path.join(OUT_DIR, f"08d_heatmap_group_{CABLE}.png")
OUT_PNG_WEIGHT = os.path.join(OUT_DIR, f"08d_heatmap_weight_{CABLE}.png")

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
ESTROUS_PHASES = ["A", "B", "C", "D"]

TARGET_WEIGHT = "body_weight"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. LOAD + BUILD PER-BAND FEATURES
# ------------------------------------------------------------
# We only need band power (05a). Ratios and oscillation-episode
# features are deliberately omitted per PROFESSOR.
# ============================================================

if not os.path.exists(BAND_POWERS_PATH):
    raise FileNotFoundError(f"Missing 05a output:\n{BAND_POWERS_PATH}")

df = pd.read_csv(BAND_POWERS_PATH)

for b in BANDS:
    df[f"log_{b}_abs"] = np.log10(df[f"{b}_abs"])

# Sanity checks
required = (["mouse", "group", "estrous_phase", TARGET_WEIGHT]
            + [f"{b}_abs" for b in BANDS]
            + [f"{b}_rel" for b in BANDS])
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"05a is missing required columns: {missing}")

# Drop recordings with no estrous phase (per-phase models cannot use them).
n_before = len(df)
df = df[df["estrous_phase"].isin(ESTROUS_PHASES)].reset_index(drop=True)
n_missing_est = n_before - len(df)

n_missing_wt = df[TARGET_WEIGHT].isna().sum()
if n_missing_wt:
    raise ValueError(
        f"{n_missing_wt} rows have no body_weight — 05a/01b should have "
        "interpolated all of these."
    )

print(f"\n{CABLE}: {len(df)} recordings kept "
      f"(dropped {n_missing_est} with no estrous phase)")
print(f"  per-phase counts: {df['estrous_phase'].value_counts().to_dict()}")


# ============================================================
# 3. MODEL / METRIC BUILDERS
# ============================================================

def make_classifier_pipes():
    """SVM-RBF + Random Forest classifiers, both class-balanced."""
    return {
        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500, class_weight="balanced",
                random_state=RANDOM_STATE)),
        ]),
    }


def make_regressor_pipes():
    """Random Forest regressor (PROFESSOR named this for weight)."""
    return {
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("reg", RandomForestRegressor(
                n_estimators=500, random_state=RANDOM_STATE)),
        ]),
    }


logo = LeaveOneGroupOut()


def eligible(sub):
    """Return True if a per-phase slice is worth modelling."""
    if len(sub) < MIN_RECORDINGS_PER_PHASE:
        return False, "too_few_recordings"
    if sub["mouse"].nunique() < MIN_MICE_PER_PHASE:
        return False, "too_few_mice"
    groups_present = sub["group"].unique()
    if len(groups_present) < 2:
        return False, "one_group_only"
    # LOMO needs at least 2 mice to leave one out.
    if sub["mouse"].nunique() < 2:
        return False, "cannot_lomo"
    return True, "ok"


def evaluate_classification(pipe, X, y, groups):
    proba = cross_val_predict(pipe, X, y, cv=logo, groups=groups,
                              method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": accuracy_score(y, pred),
    }


def evaluate_regression(pipe, X, y, groups):
    yhat = cross_val_predict(pipe, X, y, cv=logo, groups=groups)
    return {
        "r2": r2_score(y, yhat),
        "mae": mean_absolute_error(y, yhat),
    }


# ============================================================
# 4. RUN — PER PHASE, PER BAND, TWO TASKS
# ============================================================

rows = []
print("\n" + "=" * 74)
print("PER-PHASE x PER-BAND MODELS  (LOMO, group + weight)")
print("=" * 74)

for phase in ESTROUS_PHASES:
    sub = df[df["estrous_phase"] == phase].reset_index(drop=True)
    ok, reason = eligible(sub)
    n_rec = len(sub)
    n_mice_phase = sub["mouse"].nunique()
    n_hf = int((sub["group"] == POSITIVE_GROUP).sum())
    n_ctrl = int((sub["group"] != POSITIVE_GROUP).sum())

    header = (f"\n[phase {phase}]  n={n_rec}  mice={n_mice_phase}  "
              f"HF={n_hf}  CTRL={n_ctrl}")
    if not ok:
        print(header + f"  --> skipped ({reason})")
        rows.append({"phase": phase, "band": "*", "task": "*",
                     "model": "*", "metric": "skipped_reason",
                     "value": np.nan, "note": reason,
                     "n_recordings": n_rec, "n_mice": n_mice_phase})
        continue
    print(header)

    groups_arr = sub["mouse"].to_numpy()
    y_group = (sub["group"] == POSITIVE_GROUP).astype(int).to_numpy()
    y_weight = sub[TARGET_WEIGHT].to_numpy(dtype=float)

    # per-mouse HF label used for the classification permutation null
    mouse_series = sub.groupby("mouse")["group"].first()
    mouse_ids_phase = mouse_series.index.to_numpy()
    mouse_labels_phase = (mouse_series == POSITIVE_GROUP).astype(int).to_numpy()

    for band in BANDS:
        feat_cols = [f"log_{band}_abs", f"{band}_rel"]
        X = sub[feat_cols].to_numpy(dtype=float)

        # ----- CLASSIFICATION: HF vs CTRL -----
        for model_name, pipe in make_classifier_pipes().items():
            m = evaluate_classification(pipe, X, y_group, groups_arr)
            for metric, val in m.items():
                rows.append({"phase": phase, "band": band, "task": "group",
                             "model": model_name, "metric": metric,
                             "value": val, "note": "",
                             "n_recordings": n_rec, "n_mice": n_mice_phase})

        # primary group model = SVM-RBF 
        primary_group_pipe = make_classifier_pipes()["svm_rbf"]
        observed_bal = evaluate_classification(
            primary_group_pipe, X, y_group, groups_arr)["balanced_accuracy"]

        rng = np.random.default_rng(RANDOM_STATE)
        null_bal = np.empty(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            permuted = rng.permutation(mouse_labels_phase)
            mapping = dict(zip(mouse_ids_phase, permuted))
            y_perm = np.array([mapping[g] for g in groups_arr])
            if len(np.unique(y_perm)) < 2:
                null_bal[i] = 0.5
                continue
            pipe_p = make_classifier_pipes()["svm_rbf"]
            null_bal[i] = evaluate_classification(
                pipe_p, X, y_perm, groups_arr)["balanced_accuracy"]
        p_bal = (1 + np.sum(null_bal >= observed_bal)) / (1 + N_PERMUTATIONS)

        rows.append({"phase": phase, "band": band, "task": "group",
                     "model": "svm_rbf",
                     "metric": "perm_null_mean_balanced_accuracy",
                     "value": float(null_bal.mean()), "note": "",
                     "n_recordings": n_rec, "n_mice": n_mice_phase})
        rows.append({"phase": phase, "band": band, "task": "group",
                     "model": "svm_rbf",
                     "metric": "perm_p_value_balanced_accuracy",
                     "value": p_bal, "note": "",
                     "n_recordings": n_rec, "n_mice": n_mice_phase})

        # ----- REGRESSION: body weight -----
        reg_pipes = make_regressor_pipes()
        for model_name, pipe in reg_pipes.items():
            m = evaluate_regression(pipe, X, y_weight, groups_arr)
            for metric, val in m.items():
                rows.append({"phase": phase, "band": band, "task": "weight",
                             "model": model_name, "metric": metric,
                             "value": val, "note": "",
                             "n_recordings": n_rec, "n_mice": n_mice_phase})

        observed_r2 = evaluate_regression(
            make_regressor_pipes()["random_forest"], X, y_weight, groups_arr
        )["r2"]

        # weight varies per RECORDING -> shuffle at recording level
        rng = np.random.default_rng(RANDOM_STATE + 1)
        null_r2 = np.empty(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            y_perm = rng.permutation(y_weight)
            pipe_p = make_regressor_pipes()["random_forest"]
            null_r2[i] = evaluate_regression(
                pipe_p, X, y_perm, groups_arr)["r2"]
        p_r2 = (1 + np.sum(null_r2 >= observed_r2)) / (1 + N_PERMUTATIONS)

        rows.append({"phase": phase, "band": band, "task": "weight",
                     "model": "random_forest",
                     "metric": "perm_null_mean_r2",
                     "value": float(null_r2.mean()), "note": "",
                     "n_recordings": n_rec, "n_mice": n_mice_phase})
        rows.append({"phase": phase, "band": band, "task": "weight",
                     "model": "random_forest",
                     "metric": "perm_p_value_r2",
                     "value": p_r2, "note": "",
                     "n_recordings": n_rec, "n_mice": n_mice_phase})

        # small progress line so long runs don't look frozen
        g_line = [r for r in rows if r["phase"] == phase and r["band"] == band
                  and r["task"] == "group" and r["metric"] == "balanced_accuracy"]
        w_line = [r for r in rows if r["phase"] == phase and r["band"] == band
                  and r["task"] == "weight" and r["metric"] == "r2"]
        g_str = " / ".join(f"{r['model']}={r['value']:.2f}" for r in g_line)
        w_str = " / ".join(f"{r['model']}={r['value']:.2f}" for r in w_line)
        print(f"  band={band:<11s} group[bal_acc]: {g_str:<45s}"
              f"  weight[R2]: {w_str}  (p_group={p_bal:.3f}, p_weight={p_r2:.3f})")


results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved long-format results:\n{OUT_CSV}")


# ============================================================
# 5. READABLE TXT SUMMARY
# ============================================================

def pivot_metric(task, model, metric):
    """Return a phase x band table of `metric` for (task, model)."""
    sub = results_df[(results_df["task"] == task)
                     & (results_df["model"] == model)
                     & (results_df["metric"] == metric)]
    if sub.empty:
        return pd.DataFrame(index=ESTROUS_PHASES, columns=BANDS, dtype=float)
    return (sub.pivot(index="phase", columns="band", values="value")
               .reindex(index=ESTROUS_PHASES, columns=BANDS))


lines = []
lines.append("=" * 78)
lines.append(f"08D PER-PHASE x PER-BAND MODELS — {CABLE}")
lines.append("=" * 78)
lines.append("")
lines.append("Design (per PROFESSOR):")
lines.append("  * one model per (estrous phase, frequency band, task)")
lines.append("  * features per model = [log_<band>_abs, <band>_rel]  (no ratios)")
lines.append("  * validation = leave-one-mouse-out")
lines.append("  * body weight is a TARGET, never a feature")
lines.append("")
lines.append(f"Recordings kept  : {len(df)}  "
             f"(HF {(df['group']=='HF').sum()} / "
             f"CTRL {(df['group']!='HF').sum()})")
lines.append(f"Per-phase counts : {df['estrous_phase'].value_counts().to_dict()}")
lines.append(f"Eligibility rule : n >= {MIN_RECORDINGS_PER_PHASE} "
             f"and mice >= {MIN_MICE_PER_PHASE} and both groups present")
lines.append("")

for task, model_name, primary_metric, unit in [
    ("group", "svm_rbf", "balanced_accuracy", ""),
    ("group", "random_forest", "balanced_accuracy", ""),
    ("weight", "random_forest", "r2", ""),
]:
    tab = pivot_metric(task, model_name, primary_metric)
    lines.append("-" * 78)
    lines.append(f"{task.upper()}  |  model = {model_name}  |  metric = {primary_metric}")
    lines.append("-" * 78)
    lines.append(tab.round(3).fillna("--").to_string())
    lines.append("")

lines.append("-" * 78)
lines.append("PERMUTATION p-VALUES (primary models)")
lines.append("  group  : svm_rbf, balanced accuracy, mouse-level shuffle")
lines.append("  weight : random_forest, R^2, recording-level shuffle")
lines.append("-" * 78)
p_group = pivot_metric("group", "svm_rbf", "perm_p_value_balanced_accuracy")
p_weight = pivot_metric("weight", "random_forest", "perm_p_value_r2")
lines.append("group (SVM) p-values:")
lines.append(p_group.round(4).fillna("--").to_string())
lines.append("")
lines.append("weight (RF) p-values:")
lines.append(p_weight.round(4).fillna("--").to_string())
lines.append("")
lines.append("Reading guide:")
lines.append("  - Cell = one (phase, band, task) model with 2 features.")
lines.append("  - Compare bal_acc to 0.500 chance; compare R^2 to 0.")
lines.append("  - A phase-specific effect appears as one row noticeably")
lines.append("    above the others, with a small permutation p in the same cell.")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary:\n{OUT_TXT}")


# ============================================================
# 6. HEATMAPS (phase x band)
# ============================================================

def heatmap(ax, tab, title, vmin, vmax, cmap, cbar_label, ref_line=None):
    data = tab.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BANDS, rotation=30, ha="right")
    ax.set_yticks(range(len(ESTROUS_PHASES)))
    ax.set_yticklabels(ESTROUS_PHASES)
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Estrous phase")
    ax.set_title(title, fontweight="bold", loc="left")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center",
                        color="#666666", fontsize=9)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label)
    if ref_line is not None:
        cbar.ax.axhline(ref_line, color="grey", lw=1, ls="--")


# --- group classification heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
heatmap(axes[0],
        pivot_metric("group", "svm_rbf", "balanced_accuracy"),
        "A. SVM-RBF — balanced accuracy",
        vmin=0.4, vmax=0.9, cmap="viridis",
        cbar_label="balanced accuracy", ref_line=0.5)
heatmap(axes[1],
        pivot_metric("group", "random_forest", "balanced_accuracy"),
        "B. Random Forest — balanced accuracy",
        vmin=0.4, vmax=0.9, cmap="viridis",
        cbar_label="balanced accuracy", ref_line=0.5)
fig.suptitle(f"Step 08d — diet classification, phase x band  ({CABLE})",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_PNG_GROUP, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved group heatmap:\n{OUT_PNG_GROUP}")

# --- weight regression heatmap ---
fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))
heatmap(ax,
        pivot_metric("weight", "random_forest", "r2"),
        "Random Forest — R^2 (body weight)",
        vmin=-0.5, vmax=0.8, cmap="magma",
        cbar_label="R^2", ref_line=0.0)
fig.suptitle(f"Step 08d — body-weight regression, phase x band  ({CABLE})",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_PNG_WEIGHT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved weight heatmap:\n{OUT_PNG_WEIGHT}")

print("\nSTEP 08D finished successfully.")
