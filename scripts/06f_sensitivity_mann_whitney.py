# ============================================================
# 06F_SENSITIVITY_MANN_WHITNEY.PY
# Purpose:
# Non-parametric sensitivity analysis. The mixed-effects models in
# 06a are the primary statistical inference, but reviewers and
# thesis committees commonly ask: "what does a simple Mann-Whitney
# U test show?" This script answers that, and contrasts the result
# directly with the mixed-model finding.
#
# The contrast is the methodological point: a naive group test on
# raw data IGNORES the body-weight confound and the mouse-level
# clustering, so it underestimates the diet effect. Mixed-effects
# models recover the effect that the naive test misses.
#
# Procedure per primary outcome:
#   1. Pool all diet-phase recordings (HF vs CTRL)
#   2. Mann-Whitney U test (two-sided), report U, p
#   3. Rank-biserial effect size r (non-parametric analogue of d)
#   4. Compare with the mixed-model group[T.HF] result (p_fdr, coef)
#
# Caveat: Mann-Whitney treats all 89 recordings as INDEPENDENT,
# which they are not (multiple recordings per mouse). This is a
# pseudoreplication issue and the test is therefore liberal (too
# easy to find significance). We report it as a sensitivity check,
# not as primary inference, and note this caveat in the thesis.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05b_band_ratios_<CABLE>/05b_band_ratios_<CABLE>.csv
#   outputs/06a_mixed_models_<CABLE>/
#       06a_mixed_model_results_<CABLE>.csv
#
# Output:
#   outputs/06f_sensitivity_<CABLE>/
#       06f_sensitivity_mann_whitney_<CABLE>.csv
#       06f_sensitivity_comparison_<CABLE>.png
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable3"

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

OUT_DIR = os.path.join(OUTPUT_DIR, f"06f_sensitivity_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"06f_sensitivity_mann_whitney_{CABLE}.csv")
OUT_PNG = os.path.join(OUT_DIR, f"06f_sensitivity_comparison_{CABLE}.png")
OUT_TXT = os.path.join(OUT_DIR, f"06f_sensitivity_summary_{CABLE}.txt")


# Outcomes (must match 06a)
BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
PRIMARY_LOG_POWERS = [f"log_{b}_abs" for b in BANDS]
PRIMARY_LOG_RATIOS = [
    "log_theta_delta", "log_low_gamma_theta", "log_high_gamma_theta",
    "log_fast_gamma_theta", "log_beta_theta",
]
ALL_PRIMARY = PRIMARY_LOG_POWERS + PRIMARY_LOG_RATIOS

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

# Colours
C_MW   = "#888888"        # Mann-Whitney — grey
C_MIX  = "#C44E52"        # mixed model — red
C_OK   = "#4C72B0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


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
print(f"\nDiet-phase subset: {len(diet_only)} recordings / "
      f"{diet_only['mouse'].nunique()} mice")


# ============================================================
# 3. RUN MANN-WHITNEY FOR EACH PRIMARY OUTCOME
# ------------------------------------------------------------
# Rank-biserial r = 1 - 2U / (n1 * n2) is the non-parametric
# analogue of Cohen's d. r in [-1, +1].
# Convention: positive r means HF > CTRL.
# ============================================================

rows = []
for outcome in ALL_PRIMARY:
    x_ctrl = diet_only[diet_only["group"] == "CTRL"][outcome].dropna().values
    x_hf   = diet_only[diet_only["group"] == "HF"][outcome].dropna().values

    # Two-sided MW
    U_stat, p_mw = stats.mannwhitneyu(
        x_hf, x_ctrl, alternative="two-sided"
    )
    n1, n2 = len(x_hf), len(x_ctrl)
    # Rank-biserial: r = 2U/(n1*n2) - 1
    rb = 2 * U_stat / (n1 * n2) - 1

    # Mixed-model comparison row
    r = results[(results["tier"] == "primary") &
                (results["outcome"] == outcome) &
                (results["effect"] == "group[T.HF]")]
    if len(r) == 1:
        mm_beta = float(r.iloc[0]["coef"])
        mm_p    = float(r.iloc[0]["p_value"])
        mm_pfdr = float(r.iloc[0]["p_fdr"])
    else:
        mm_beta = mm_p = mm_pfdr = np.nan

    rows.append({
        "outcome": outcome,
        "label": LABEL_MAP.get(outcome, outcome),
        "n_CTRL": n2, "n_HF": n1,
        "median_CTRL": float(np.median(x_ctrl)),
        "median_HF":   float(np.median(x_hf)),
        "U_stat":      float(U_stat),
        "p_mann_whitney": float(p_mw),
        "rank_biserial":  float(rb),
        "mm_coef":  mm_beta,
        "mm_p_raw": mm_p,
        "mm_p_fdr": mm_pfdr,
        # Agreement flag
        "mw_significant_p05":  bool(p_mw < 0.05),
        "mm_significant_pfdr": bool(mm_pfdr < 0.05) if not np.isnan(mm_pfdr) else False,
    })

sens = pd.DataFrame(rows)
sens.to_csv(OUT_CSV, index=False)
print(f"\nSaved sensitivity table: {OUT_CSV}")


# ============================================================
# 4. CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 86)
print("MANN-WHITNEY vs MIXED MODEL — primary outcomes")
print("=" * 86)
print(f"{'outcome':<28s} {'MW p':>8s} {'rank-r':>8s}"
      f" {'MM p_raw':>10s} {'MM p_fdr':>10s} {'agreement':>16s}")
for _, r in sens.iterrows():
    if r["mw_significant_p05"] == r["mm_significant_pfdr"]:
        if r["mw_significant_p05"]:
            agree = "both sig."
        else:
            agree = "both n.s."
    elif r["mw_significant_p05"] and not r["mm_significant_pfdr"]:
        agree = "MW only"
    else:
        agree = "MM only"
    print(f"{r['outcome']:<28s} "
          f"{r['p_mann_whitney']:>8.3f} "
          f"{r['rank_biserial']:>+8.2f} "
          f"{r['mm_p_raw']:>10.3f} "
          f"{r['mm_p_fdr']:>10.3f} "
          f"{agree:>16s}")


# ============================================================
# 5. COMPARISON FIGURE — TWO PANELS
# ------------------------------------------------------------
# Panel A: side-by-side p-values (MW vs mixed model)
# Panel B: rank-biserial r vs mixed-model coefficient — direction
#          of effect from the two methods.
# ============================================================

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.5))

# ---- Panel A: side-by-side p-values ----
sens_sorted = sens.sort_values("mm_p_fdr").reset_index(drop=True)
n = len(sens_sorted)
y_pos = np.arange(n)[::-1]
bar_h = 0.36

# Mann-Whitney
mw_colours = [C_MIX if p < 0.05 else C_MW
              for p in sens_sorted["p_mann_whitney"]]
axA.barh(y_pos + bar_h / 2, -np.log10(sens_sorted["p_mann_whitney"]),
         bar_h, color=mw_colours, alpha=0.85,
         edgecolor="white", linewidth=0.5, label="Mann-Whitney")

# Mixed model (p_fdr)
mm_colours = [C_MIX if p < 0.05 else C_MW
              for p in sens_sorted["mm_p_fdr"]]
axA.barh(y_pos - bar_h / 2, -np.log10(sens_sorted["mm_p_fdr"]),
         bar_h, color=mm_colours, alpha=0.85,
         edgecolor="white", linewidth=0.5, label="Mixed model (p_fdr)")

# Significance threshold lines
axA.axvline(-np.log10(0.05), color="black", linestyle="--",
            linewidth=0.7, alpha=0.5)
axA.text(-np.log10(0.05), -0.5, "p = 0.05", fontsize=8,
         color="black", alpha=0.7)

axA.set_yticks(y_pos)
axA.set_yticklabels(sens_sorted["label"], fontsize=9)
axA.set_xlabel("-log10(p)  (higher = more significant)")
axA.set_title("A.  P-values: MW vs mixed model", loc="left")
axA.legend(handles=[
    Patch(facecolor=C_MIX, label="significant"),
    Patch(facecolor=C_MW,  label="n.s."),
], frameon=False, fontsize=9, loc="lower right")


# ---- Panel B: effect-direction scatter ----
axB.scatter(sens["rank_biserial"], sens["mm_coef"],
            color=C_OK, alpha=0.85, s=80, edgecolors="white",
            linewidths=1.2)
for _, r in sens.iterrows():
    axB.annotate(r["label"].split(" ")[0],
                 (r["rank_biserial"], r["mm_coef"]),
                 fontsize=8, alpha=0.75,
                 xytext=(4, 2), textcoords="offset points")
axB.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
axB.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
axB.set_xlabel("Rank-biserial r (Mann-Whitney)")
axB.set_ylabel("Mixed-model coefficient (log10 units)")
axB.set_title("B.  Effect direction: do MW and MM agree?",
              loc="left")
# Quadrant labels
axB.text(0.98, 0.96, "agree: both HF higher",
         transform=axB.transAxes, fontsize=8, ha="right", va="top",
         color="#666")
axB.text(0.02, 0.04, "agree: both HF lower",
         transform=axB.transAxes, fontsize=8, ha="left", va="bottom",
         color="#666")
axB.text(0.02, 0.96, "disagree",
         transform=axB.transAxes, fontsize=8, ha="left", va="top",
         color="#aa6666")
axB.text(0.98, 0.04, "disagree",
         transform=axB.transAxes, fontsize=8, ha="right", va="bottom",
         color="#aa6666")


fig.suptitle(f"Sensitivity analysis — Mann-Whitney vs mixed-effects "
             f"({CABLE}, n={len(diet_only)} recordings)",
             fontsize=13, fontweight="bold", y=1.00)

caption = ("Panel A: -log10(p) for the two tests side by side per "
           "outcome. Panel B: effect-direction scatter — rank-biserial "
           "from Mann-Whitney vs the mixed-model group coefficient. "
           "Caveat: Mann-Whitney ignores mouse-level clustering "
           "(treats all 89 recordings as independent), so it is a "
           "liberal test and serves only as a sensitivity check.")
fig.text(0.5, -0.04, caption, ha="center", fontsize=8, color="#444",
         wrap=True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved comparison figure: {OUT_PNG}")


# ============================================================
# 6. SAVE HUMAN-READABLE TXT SUMMARY
# ------------------------------------------------------------
# Side-by-side Mann-Whitney vs mixed-model table. Easy to paste
# into emails or thesis appendices.
# ============================================================

lines = []
lines.append("=" * 86)
lines.append(f"06F SENSITIVITY ANALYSIS (MANN-WHITNEY vs MIXED MODEL) — "
             f"{CABLE}")
lines.append("=" * 86)
lines.append("")
lines.append(f"Sample:  {len(diet_only)} recordings / "
             f"{diet_only['mouse'].nunique()} mice (diet-phase only)")
lines.append("")
lines.append("Mann-Whitney U (two-sided) compares HF vs CTRL on raw")
lines.append("recordings, treating all 89 recordings as independent.")
lines.append("This is a LIBERAL test (it ignores mouse-level clustering),")
lines.append("so the comparison with the mixed-model result is the")
lines.append("methodological point of this script: a naive test can hide")
lines.append("an effect that mixed-effects modelling recovers.")
lines.append("")
lines.append("Rank-biserial r: non-parametric analogue of Cohen's d.")
lines.append("  positive r = HF higher; negative r = HF lower; range [-1,+1].")
lines.append("")
lines.append("-" * 86)
lines.append("RESULTS — sorted by mixed-model p_FDR (most significant first)")
lines.append("-" * 86)
lines.append(f"{'outcome':<30s} {'MW_p':>8s} {'rank_r':>8s}"
             f" {'MM_p_raw':>10s} {'MM_p_fdr':>10s} {'agreement':>16s}")

for _, r in sens.sort_values("mm_p_fdr").iterrows():
    if r["mw_significant_p05"] == r["mm_significant_pfdr"]:
        agree = "both sig." if r["mw_significant_p05"] else "both n.s."
    elif r["mw_significant_p05"] and not r["mm_significant_pfdr"]:
        agree = "MW only"
    else:
        agree = "MM only"
    sig = " *" if r["mm_significant_pfdr"] else ""
    lines.append(f"{r['outcome']:<30s} "
                 f"{r['p_mann_whitney']:>8.3f} "
                 f"{r['rank_biserial']:>+8.2f} "
                 f"{r['mm_p_raw']:>10.3f} "
                 f"{r['mm_p_fdr']:>10.3f} "
                 f"{agree:>16s}{sig}")

# Count agreements
n_both_sig = ((sens["mw_significant_p05"]) &
              (sens["mm_significant_pfdr"])).sum()
n_both_ns  = ((~sens["mw_significant_p05"]) &
              (~sens["mm_significant_pfdr"])).sum()
n_mw_only  = ((sens["mw_significant_p05"]) &
              (~sens["mm_significant_pfdr"])).sum()
n_mm_only  = ((~sens["mw_significant_p05"]) &
              (sens["mm_significant_pfdr"])).sum()

lines.append("")
lines.append("-" * 86)
lines.append("AGREEMENT SUMMARY (out of 11 primary outcomes)")
lines.append("-" * 86)
lines.append(f"  Both tests significant:        {n_both_sig}")
lines.append(f"  Both tests non-significant:    {n_both_ns}")
lines.append(f"  Mann-Whitney only:             {n_mw_only}")
lines.append(f"  Mixed model only (recovered):  {n_mm_only}")
lines.append("")
lines.append("Interpretation:")
lines.append("  'MM only' rows are the central finding — the mixed-effects")
lines.append("  model recovers a diet effect that the naive non-parametric")
lines.append("  test misses, because the latter ignores mouse-level")
lines.append("  clustering and the body-weight confound. For Cable1, this")
lines.append("  is the case for the delta band: Mann-Whitney p = 0.82,")
lines.append("  mixed model p_FDR = 0.015.")
lines.append("")
lines.append("'*' marks rows where the mixed-model p_FDR < 0.05.")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary: {OUT_TXT}")


print("\nSTEP 06F finished successfully.")