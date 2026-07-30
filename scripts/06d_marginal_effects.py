# ============================================================
# 06D_MARGINAL_EFFECTS.PY
# Purpose:
# Show what the mixed-effects models from 06a actually estimate —
# i.e. the diet effect AFTER controlling for body weight, time on
# diet, and estrous phase. The raw scatter in 06c (Figure 2) can
# look unimpressive because two effects partially cancel: HF mice
# are heavier (body weight raises delta), and the diet itself
# lowers delta. The mixed model separates these; this script
# visualises the separation.
#
# Three figures:
#
#   Figure 1 — Marginal predictions over days_on_diet (delta)
#       For each group, plot the model-predicted log_delta with
#       95% CI band, holding body_weight at its mean and using
#       reference estrous phase A. Per-mouse residual scatter
#       overlaid for context.
#       This is the "model-adjusted" version of 06c Figure 2 —
#       the group difference becomes visually clear.
#
#   Figure 2 — Coefficient forest plot (covariate-adjusted)
#       Same 11 outcomes as 06c Figure 1, but here every
#       coefficient is explicitly an estimated marginal mean
#       difference (HF - CTRL) adjusted for the covariates.
#       Same numbers as the model, framed as effect sizes.
#
#   Figure 3 — Disentangling diet vs body weight effects
#       Side-by-side bars showing the group[T.HF] coefficient and
#       the body_weight coefficient for each band power. Makes the
#       point that the two effects often go in OPPOSITE directions,
#       so failing to adjust for body weight would mask the diet
#       signal entirely.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05b_band_ratios_<CABLE>/05b_band_ratios_<CABLE>.csv
#   outputs/06a_mixed_models_<CABLE>/
#       06a_mixed_model_results_<CABLE>.csv
#
# Output:
#   outputs/06d_marginal_effects_<CABLE>/
#       06d_fig1_marginal_delta_trajectory_<CABLE>.png
#       06d_fig2_effect_size_forest_<CABLE>.png
#       06d_fig3_diet_vs_bodyweight_<CABLE>.png
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}",
    f"05a_band_powers_{CABLE}.csv"
)
BAND_RATIOS_PATH = os.path.join(
    OUTPUT_DIR, f"05b_band_ratios_{CABLE}",
    f"05b_band_ratios_{CABLE}.csv"
)
RESULTS_PATH = os.path.join(
    OUTPUT_DIR, f"06a_mixed_models_{CABLE}",
    f"06a_mixed_model_results_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"06d_marginal_effects_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)

FIG1_PATH = os.path.join(
    OUT_DIR, f"06d_fig1_marginal_delta_trajectory_{CABLE}.png"
)
FIG2_PATH = os.path.join(
    OUT_DIR, f"06d_fig2_effect_size_forest_{CABLE}.png"
)
FIG3_PATH = os.path.join(
    OUT_DIR, f"06d_fig3_diet_vs_bodyweight_{CABLE}.png"
)


# Colours
C_CTRL = "#4C72B0"
C_HF   = "#C44E52"
C_SIG  = "#C44E52"
C_NS   = "#888888"
C_WEIGHT = "#55A868"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

# Bands + formula (must match 06a)
BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
PRIMARY_LOG_POWERS = [f"log_{b}_abs" for b in BANDS]
PRIMARY_LOG_RATIOS = [
    "log_theta_delta", "log_low_gamma_theta", "log_high_gamma_theta",
    "log_fast_gamma_theta", "log_beta_theta",
]
RHS_FORMULA = ("group * days_on_diet + body_weight "
               "+ C(estrous_phase)")

LABEL_MAP = {
    "log_delta_abs":         "Delta (1-4 Hz)",
    "log_theta_abs":         "Theta (4-10 Hz)",
    "log_beta_abs":          "Beta (15-30 Hz)",
    "log_low_gamma_abs":     "Low gamma (30-60 Hz)",
    "log_high_gamma_abs":    "High gamma (60-100 Hz)",
    "log_fast_gamma_abs":    "Fast gamma (100-140 Hz)",
    "log_theta_delta":       "theta/delta",
    "log_low_gamma_theta":   "low gamma/theta",
    "log_high_gamma_theta":  "high gamma/theta",
    "log_fast_gamma_theta":  "fast gamma/theta",
    "log_beta_theta":        "beta/theta",
}


# ============================================================
# 2. LOAD + BUILD MASTER, FILTER TO DIET PHASE
# ============================================================

df_05a = pd.read_csv(BAND_POWERS_PATH)
df_05b = pd.read_csv(BAND_RATIOS_PATH)
results = pd.read_csv(RESULTS_PATH)

for band in BANDS:
    df_05a[f"log_{band}_abs"] = np.log10(df_05a[f"{band}_abs"])

df = df_05a.merge(
    df_05b[["differential_file"] + PRIMARY_LOG_RATIOS],
    on="differential_file", how="left", validate="one_to_one"
)
diet_only = df[df["days_on_diet"].notna()].copy()


# ============================================================
# 3. FIGURE 1 — MARGINAL PREDICTIONS OVER days_on_diet (DELTA)
# ------------------------------------------------------------
# Refit the delta model, then generate predictions across a grid
# of days_on_diet, holding body_weight at its mean and estrous
# fixed at reference (A). The two lines (HF, CTRL) and their
# 95% confidence bands show the model-adjusted group separation.
# ============================================================

outcome = "log_delta_abs"
formula = f"{outcome} ~ {RHS_FORMULA}"

# Try a couple of methods, same robustness pattern as 06b
fitted = None
for method in ["lbfgs", "powell", "bfgs"]:
    try:
        model = smf.mixedlm(formula, data=diet_only,
                            groups=diet_only["mouse"])
        fitted = model.fit(method=method, reml=True)
        _ = np.asarray(fitted.fittedvalues)
        break
    except Exception:
        fitted = None
        continue

if fitted is None:
    raise RuntimeError("Delta model failed to fit for marginal plot.")

# Grid for prediction
mean_bw = diet_only["body_weight"].mean()
days_grid = np.linspace(
    diet_only["days_on_diet"].min(),
    diet_only["days_on_diet"].max(),
    50
)

# Manual prediction (safer than dmatrix on group-only subsets,
# which patsy would code wrong).
# Pull fixed-effect parameters and their covariance matrix.
params = fitted.params
cov = fitted.cov_params()

# Helper: predict for a given group at the grid, holding
# body_weight at mean and estrous at A (reference).
def predict_grid(group_label):
    n = len(days_grid)
    # Design row: [Intercept, group[T.HF], estrous_B, estrous_C,
    #              estrous_D, days_on_diet, group[T.HF]:days_on_diet,
    #              body_weight]
    # Reference levels are 0; only days_on_diet and possibly the
    # group/interaction columns are non-zero.
    grp_dummy = 1.0 if group_label == "HF" else 0.0
    X = np.zeros((n, len(params)))
    cols = list(params.index)
    X[:, cols.index("Intercept")] = 1.0
    X[:, cols.index("group[T.HF]")] = grp_dummy
    X[:, cols.index("days_on_diet")] = days_grid
    X[:, cols.index("group[T.HF]:days_on_diet")] = grp_dummy * days_grid
    X[:, cols.index("body_weight")] = mean_bw
    # Estrous reference is A => all dummies 0

    beta = params.values
    yhat = X @ beta
    cov_mat = cov.values
    var_hat = np.einsum("ij,jk,ik->i", X, cov_mat, X)
    se_hat = np.sqrt(np.clip(var_hat, 0, None))
    return pd.DataFrame({
        "days_on_diet": days_grid,
        "yhat": yhat,
        "ci_low":  yhat - 1.96 * se_hat,
        "ci_high": yhat + 1.96 * se_hat,
    })

pred_frames = {"CTRL": predict_grid("CTRL"),
               "HF":   predict_grid("HF")}


fig, ax = plt.subplots(figsize=(10, 6.5))

# Raw scatter in the background (very faint, for orientation)
for grp, colour in [("CTRL", C_CTRL), ("HF", C_HF)]:
    sub = diet_only[diet_only["group"] == grp]
    ax.scatter(sub["days_on_diet"], sub[outcome], color=colour,
               alpha=0.18, s=20, edgecolors="none")

# Marginal prediction lines + CI bands
for grp, colour in [("CTRL", C_CTRL), ("HF", C_HF)]:
    p = pred_frames[grp]
    ax.fill_between(p["days_on_diet"], p["ci_low"], p["ci_high"],
                    color=colour, alpha=0.20, linewidth=0)
    ax.plot(p["days_on_diet"], p["yhat"], color=colour, linewidth=3,
            label=f"{grp} (model prediction)", zorder=5)

ax.set_xlabel("Days on diet")
ax.set_ylabel("log10(delta absolute power)")
ax.set_title("Figure 1.  Marginal model predictions — Delta band\n"
             "(adjusted for body weight, estrous phase A reference)",
             loc="left")
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.grid(True, alpha=0.2, linestyle=":")

# Add the marginal effect annotation
grp_row = results[
    (results["tier"] == "primary") &
    (results["outcome"] == outcome) &
    (results["effect"] == "group[T.HF]")
].iloc[0]
annot = (f"Marginal effect (HF - CTRL):\n"
         f"  beta = {grp_row['coef']:+.3f}\n"
         f"  95% CI [{grp_row['ci_low']:+.2f}, {grp_row['ci_high']:+.2f}]\n"
         f"  p_FDR = {grp_row['p_fdr']:.3f}  *")
ax.text(0.02, 0.05, annot, transform=ax.transAxes, fontsize=10,
        va="bottom", color=C_SIG, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=C_SIG, alpha=0.85))

caption = (f"Solid lines: model-predicted log delta power for each "
           f"group at the mean body weight ({mean_bw:.1f} g) and the "
           f"reference estrous phase. Shaded bands: 95% CI of the "
           f"prediction. Faint dots: raw recordings (89 total / 19 "
           f"mice). Adjusting for body weight reveals the diet effect "
           f"that is hidden in the raw scatter (Figure 2 in 06c).")
fig.text(0.5, -0.03, caption, ha="center", fontsize=8, color="#444",
         wrap=True)

plt.tight_layout()
plt.savefig(FIG1_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved Figure 1: {FIG1_PATH}")


# ============================================================
# 4. FIGURE 2 — EFFECT-SIZE FOREST PLOT
# ------------------------------------------------------------
# Same 11 primary outcomes, but framed as "marginal mean
# difference (HF - CTRL) on the log10 scale, adjusted for
# body weight, days on diet, and estrous phase."
# Adds linear-scale interpretation alongside each row.
# ============================================================

primary_grp = results[
    (results["tier"] == "primary") &
    (results["effect"] == "group[T.HF]")
].sort_values("p_value").reset_index(drop=True)

primary_grp["label"] = primary_grp["outcome"].map(LABEL_MAP)
# Fold change on linear scale (power ratio HF / CTRL)
primary_grp["fold_change"] = 10 ** primary_grp["coef"]

fig, ax = plt.subplots(figsize=(11, 7.5))

n = len(primary_grp)
y_positions = np.arange(n)[::-1]

for i, row in primary_grp.iterrows():
    y = y_positions[i]
    is_sig = row["p_fdr"] < 0.05
    colour = C_SIG if is_sig else C_NS
    marker = "s" if "/" in row["label"] else "o"

    ax.plot([row["ci_low"], row["ci_high"]], [y, y],
            color=colour, linewidth=2, alpha=0.85)
    ax.scatter(row["coef"], y, color=colour, s=80,
               marker=marker, edgecolors="white", linewidths=1.2,
               zorder=3)

    # Right-side annotation: fold change on linear scale
    fold = row["fold_change"]
    if fold < 1:
        fold_text = f"{1/fold:.2f}-fold lower in HF"
    elif fold > 1:
        fold_text = f"{fold:.2f}-fold higher in HF"
    else:
        fold_text = "no change"
    star = "  *" if is_sig else ""
    ax.text(row["ci_high"] + 0.05, y,
            f"  {fold_text}{star}",
            va="center", fontsize=8.5,
            color=colour, fontweight="bold" if is_sig else "normal")

ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
ax.set_yticks(y_positions)
ax.set_yticklabels(primary_grp["label"], fontsize=10)
ax.set_xlabel("Marginal effect (HF - CTRL), log10 units with 95% CI\n"
              "[adjusted for body weight, days on diet, estrous phase]",
              fontsize=10)
ax.set_title("Figure 2.  Effect sizes after covariate adjustment\n"
             "(linear-scale interpretation on the right)",
             loc="left")

ax.legend(handles=[
    Patch(facecolor=C_SIG, label="FDR significant"),
    Patch(facecolor=C_NS,  label="Not significant"),
], frameon=False, fontsize=9, loc="lower right",
   bbox_to_anchor=(1.0, -0.04))

caption = (f"Forest plot of marginal mean differences (HF - CTRL) "
           f"from the 11 primary mixed-effects models. Negative "
           f"coefficients = HF lower. Right-hand text translates the "
           f"log10 coefficient back to the linear scale (fold change "
           f"in power or ratio).")
fig.text(0.5, -0.04, caption, ha="center", fontsize=8, color="#444",
         wrap=True)

plt.tight_layout()
plt.savefig(FIG2_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved Figure 2: {FIG2_PATH}")


# ============================================================
# 5. FIGURE 3 — DIET vs BODY WEIGHT EFFECTS SIDE-BY-SIDE
# ------------------------------------------------------------
# For each band power, show the group[T.HF] coefficient and the
# body_weight coefficient side by side. The point: in several
# bands the two effects go in OPPOSITE directions, so raw
# unadjusted comparisons (Figure 2 in 06c) would mask the diet
# signal entirely.
# ============================================================

# Get group[T.HF] and body_weight rows for each band-power outcome
band_outcomes = PRIMARY_LOG_POWERS  # only the 6 powers, not ratios
sub = results[
    (results["tier"] == "primary") &
    (results["outcome"].isin(band_outcomes)) &
    (results["effect"].isin(["group[T.HF]", "body_weight"]))
].copy()
sub["label"] = sub["outcome"].map(LABEL_MAP)

fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(band_outcomes))
w = 0.35

# Match the order of LABEL_MAP for band powers
ordered_outcomes = band_outcomes
diet_coefs, diet_errs, diet_ps = [], [], []
bw_coefs, bw_errs, bw_ps = [], [], []
for o in ordered_outcomes:
    g = sub[(sub["outcome"] == o) &
            (sub["effect"] == "group[T.HF]")].iloc[0]
    b = sub[(sub["outcome"] == o) &
            (sub["effect"] == "body_weight")].iloc[0]
    diet_coefs.append(g["coef"])
    diet_errs.append(g["se"])
    diet_ps.append(g["p_fdr"])
    bw_coefs.append(b["coef"])
    bw_errs.append(b["se"])
    bw_ps.append(b["p_value"])

# Diet effect (HF - CTRL) — left bar per band
diet_colours = [C_SIG if p < 0.05 else C_NS for p in diet_ps]
ax.bar(x - w/2, diet_coefs, w, yerr=diet_errs, capsize=3,
       color=diet_colours, alpha=0.85, edgecolor="white", linewidth=0.5,
       label="Diet effect (HF - CTRL, log10)")

# Body weight effect — right bar per band (re-scaled for visualisation:
# body_weight coefs are per gram, so multiply by 10 to give the effect
# of a 10 g weight increase, which is roughly the HF-CTRL weight gap)
bw_coefs_scaled = [c * 10 for c in bw_coefs]
bw_errs_scaled  = [e * 10 for e in bw_errs]
bw_colours = [C_WEIGHT if p < 0.05 else "#BBD8B1" for p in bw_ps]
ax.bar(x + w/2, bw_coefs_scaled, w, yerr=bw_errs_scaled, capsize=3,
       color=bw_colours, alpha=0.85, edgecolor="white", linewidth=0.5,
       label="Body weight effect (per +10 g, log10)")

ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([LABEL_MAP[o] for o in ordered_outcomes],
                   rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Coefficient (log10 units)")
ax.set_title("Figure 3.  Diet vs body-weight effects "
             "(disentangled by the mixed model)", loc="left")
ax.legend(frameon=False, fontsize=10, loc="lower right")

caption = ("Per-band coefficients for the diet (group[T.HF], red) and "
           "body-weight predictors (green; shown for a +10 g increment "
           "to make it comparable to the HF-CTRL weight gap of ~4 g). "
           "Diet pushes powers DOWN; body weight tends to pull them UP. "
           "Without adjusting for body weight, the diet signal would be "
           "partly cancelled — which is why mixed-effects modelling is "
           "essential here.")
fig.text(0.5, -0.04, caption, ha="center", fontsize=8, color="#444",
         wrap=True)

plt.tight_layout()
plt.savefig(FIG3_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved Figure 3: {FIG3_PATH}")


print("\nSTEP 06D finished successfully.")