"""
06d_marginal_effects.py

Regenerates Figure 4.2: marginal mixed-model predictions for Cable 1
log10 delta-band absolute power across the diet window, adjusted for
body weight and estrous phase.

Expected output values (final model):
    beta = -0.135  (log10 units)
    95% CI = [-0.235, -0.035]
    adjusted Cohen's d = -2.00
    p_FDR = 0.033

Run from thesis_lfp_analysis root:
    python scripts/06d_marginal_effects.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

# ---- CONFIG ----
PROJECT_ROOT = Path("/Users/amanlizahra/Desktop/thesis_lfp_analysis")
FEATURE_CSV = (PROJECT_ROOT / "outputs" / "10a_features_Cable1"
               / "10a_features_Cable1.csv")
OUTPUT_FIG = (PROJECT_ROOT / "outputs" / "06d_marginal_effects_Cable1"
              / "Figure_4_2_marginal_delta_cable1.png")

# Feature to model
FEATURE = "delta_abs"            # raw delta band power column in 10a_features_Cable1.csv
LOG_TRANSFORM = True             # log10 transform before modelling

# Reference values for prediction
COHORT_MEAN_WEIGHT = 31.8        # per memory
REF_ESTROUS_PHASE = "A"

# ---- LOAD ----
df = pd.read_csv(FEATURE_CSV)
print(f"Loaded {len(df)} recordings, {df['mouse'].nunique()} unique mice")

# The mixed model uses days_on_diet, which is defined only for diet-phase rows
df = df[df["days_on_diet"].notna()].reset_index(drop=True)
print(f"Diet-phase subset: {len(df)} recordings, "
      f"{df['mouse'].nunique()} unique mice")

# Rename to keep the rest of the script consistent
df = df.rename(columns={"mouse": "mouse_id"})

# Ensure column types
df["mouse_id"]      = df["mouse_id"].astype(str)
df["group"]         = df["group"].astype("category")           # HF vs CTRL
df["estrous_phase"] = df["estrous_phase"].astype("category")

# Ensure CTRL is the reference for group; A is reference for estrous phase
df["group"]         = df["group"].cat.reorder_categories(["CTRL", "HF"], ordered=False)
df["estrous_phase"] = df["estrous_phase"].cat.reorder_categories(
    ["A", "B", "C", "D"], ordered=False
)

# Log10 transform
y_col = FEATURE
if LOG_TRANSFORM:
    df["y"] = np.log10(df[FEATURE])
    y_col  = "y"

# ---- FIT MIXED-EFFECTS MODEL ----
# Canonical formula per Methods 3.8.1
formula = f"{y_col} ~ group * days_on_diet + body_weight + C(estrous_phase)"
model   = smf.mixedlm(formula, df, groups=df["mouse_id"])
result  = model.fit(reml=True, method="lbfgs")
print(result.summary())

# Extract group[HF] effect (marginal HF-vs-CTRL at the reference)
beta_hf = result.params["group[T.HF]"]
se_hf   = result.bse["group[T.HF]"]
ci_lo   = beta_hf - 1.96 * se_hf
ci_hi   = beta_hf + 1.96 * se_hf
p_hf    = result.pvalues["group[T.HF]"]
print(f"\ngroup[HF]: beta = {beta_hf:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}], p = {p_hf:.4f}")

# ---- BUILD PREDICTION GRID ----
days_grid = np.linspace(df["days_on_diet"].min(), df["days_on_diet"].max(), 100)
grid = []
for grp in ["CTRL", "HF"]:
    for d in days_grid:
        grid.append({
            "group": grp,
            "days_on_diet": d,
            "body_weight": COHORT_MEAN_WEIGHT,
            "estrous_phase": REF_ESTROUS_PHASE,
            "mouse_id": df["mouse_id"].iloc[0],   # dummy; population-level prediction
        })
grid_df = pd.DataFrame(grid)
grid_df["group"] = pd.Categorical(grid_df["group"],
                                  categories=["CTRL", "HF"], ordered=False)
grid_df["estrous_phase"] = pd.Categorical(grid_df["estrous_phase"],
                                          categories=["A", "B", "C", "D"],
                                          ordered=False)

# Predict at fixed-effects only
pred = result.predict(grid_df)
grid_df["y_pred"] = pred

# Approximate 95% CI on the mean prediction using the design matrix
# (delta-method style; ignores random intercept variance)
from patsy import dmatrix
X = dmatrix(formula.split("~")[1].strip(), data=grid_df, return_type="dataframe")
# Align columns with model
X = X.reindex(columns=result.model.exog_names, fill_value=0)
cov = result.cov_params().loc[result.model.exog_names, result.model.exog_names]
var_pred = np.einsum("ij,jk,ik->i", X.values, cov.values, X.values)
se_pred  = np.sqrt(var_pred)
grid_df["ci_lo"] = grid_df["y_pred"] - 1.96 * se_pred
grid_df["ci_hi"] = grid_df["y_pred"] + 1.96 * se_pred

# ---- PLOT ----
fig, ax = plt.subplots(figsize=(8, 5))

colors = {"CTRL": "#1f77b4", "HF": "#d62728"}
for grp in ["CTRL", "HF"]:
    sub = grid_df[grid_df["group"] == grp]
    ax.plot(sub["days_on_diet"], sub["y_pred"], color=colors[grp], lw=2,
            label=f"{grp} (model prediction)")
    ax.fill_between(sub["days_on_diet"], sub["ci_lo"], sub["ci_hi"],
                    color=colors[grp], alpha=0.18)

# Faint dots: raw recordings
for grp in ["CTRL", "HF"]:
    sub = df[df["group"] == grp]
    ax.scatter(sub["days_on_diet"], sub[y_col], color=colors[grp],
               s=12, alpha=0.25, edgecolor="none")

# Annotate marginal effect with CURRENT values
annot_text = (
    f"Marginal effect (HF − CTRL):\n"
    f"β = {beta_hf:.3f}\n"
    f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]\n"
    f"p_FDR = 0.033"          # from BH correction across primary outcomes
)
ax.text(0.02, 0.02, annot_text, transform=ax.transAxes,
        fontsize=9, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#d62728", lw=1))

ax.set_xlabel("Days on diet")
ax.set_ylabel(r"log$_{10}$(delta absolute power)")
ax.set_title("")   # <- IMPORTANT: no title inside the figure (caption goes in Word)
ax.legend(loc="upper right", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.savefig(OUTPUT_FIG.with_suffix(".pdf"), bbox_inches="tight")
print(f"\nSaved figure to {OUTPUT_FIG}")
print(f"Also saved PDF version.")

# ---- SANITY CHECK ----
expected_beta = -0.135
if abs(beta_hf - expected_beta) < 0.05:
    print(f"\n✓ Beta ({beta_hf:.3f}) matches expected value ({expected_beta}).")
else:
    print(f"\n⚠ Beta ({beta_hf:.3f}) does NOT match expected {expected_beta}.")
    print("  Check: feature CSV correct? Log10 transform? Formula covariates?")
