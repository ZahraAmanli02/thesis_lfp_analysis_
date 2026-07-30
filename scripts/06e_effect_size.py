# ============================================================
# 06E_EFFECT_SIZES.PY
# Purpose:
# Compute Cohen's d effect sizes for the 11 primary outcomes
# from 06a. P-values tell us "is there a difference?"; Cohen's
# d tells us "how big is the difference?". The two together give
# a much richer interpretation than either alone, and Cohen's d
# is now standard in any thesis or publication.
#
# We compute TWO versions:
#
#   1. Classical Cohen's d (raw):
#         d = (mean_HF - mean_CTRL) / pooled_SD
#      Uses raw data, does NOT adjust for covariates.
#      Pros: easy to interpret, matches what most readers expect.
#      Cons: ignores body_weight, days_on_diet, estrous, mouse
#            clustering — so it reports the "marginal" difference
#            including the body-weight confound.
#
#   2. Mixed-model adjusted Cohen's d:
#         d_adj = beta_group / sqrt(var_mouse + var_residual)
#      Uses the group[T.HF] coefficient from 06a directly,
#      standardised by the total model variance.
#      This is the "pure diet effect" after controlling for
#      body weight, days on diet, and estrous phase.
#
# Both are reported. The thesis can cite either, but the adjusted
# version aligns with the mixed-model story and is more honest
# about the body-weight confound.
#
# Cohen (1988) interpretation:
#   |d| < 0.2   negligible
#   0.2 - 0.5   small
#   0.5 - 0.8   medium
#   0.8 - 1.2   large
#   > 1.2       very large
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05b_band_ratios_<CABLE>/05b_band_ratios_<CABLE>.csv
#   outputs/06a_mixed_models_<CABLE>/
#       06a_mixed_model_results_<CABLE>.csv
#
# Output:
#   outputs/06e_effect_sizes_<CABLE>/
#       06e_effect_sizes_<CABLE>.csv
#       06e_effect_sizes_<CABLE>.png
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

OUT_DIR = os.path.join(OUTPUT_DIR, f"06e_effect_sizes_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"06e_effect_sizes_{CABLE}.csv")
OUT_PNG = os.path.join(OUT_DIR, f"06e_effect_sizes_{CABLE}.png")
OUT_TXT = os.path.join(OUT_DIR, f"06e_effect_sizes_summary_{CABLE}.txt")


# Bands
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

# Cohen's d interpretation thresholds
COHEN_THRESHOLDS = [
    (0.2,  "negligible"),
    (0.5,  "small"),
    (0.8,  "medium"),
    (1.2,  "large"),
    (10.0, "very large"),
]


# Colours
C_CTRL = "#4C72B0"
C_HF   = "#C44E52"
C_SIG  = "#C44E52"
C_NS   = "#888888"

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
# 3. COHEN'S D HELPERS
# ============================================================

def classical_cohens_d(x_ctrl, x_hf):
    """
    Classical Cohen's d using pooled standard deviation.
    Positive d means HF > CTRL on this outcome.
    """
    x_ctrl = np.asarray(x_ctrl, dtype=float)
    x_hf   = np.asarray(x_hf,   dtype=float)
    x_ctrl = x_ctrl[~np.isnan(x_ctrl)]
    x_hf   = x_hf[~np.isnan(x_hf)]
    n1, n2 = len(x_ctrl), len(x_hf)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    var1 = np.var(x_ctrl, ddof=1)
    var2 = np.var(x_hf,   ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2)
                        / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan, pooled_sd
    d = (np.mean(x_hf) - np.mean(x_ctrl)) / pooled_sd
    return float(d), float(pooled_sd)


def adjusted_cohens_d(beta, var_mouse, var_resid):
    """
    Mixed-model-adjusted Cohen's d: the diet effect from the model
    standardised by total model variance (mouse + residual).
    """
    total_sd = np.sqrt(var_mouse + var_resid)
    if total_sd == 0:
        return np.nan
    return float(beta / total_sd)


def cohen_label(d):
    """Map |d| to a verbal magnitude tag."""
    ad = abs(d)
    if np.isnan(ad):
        return "n/a"
    for threshold, name in COHEN_THRESHOLDS:
        if ad < threshold:
            return name
    return "very large"


# ============================================================
# 4. COMPUTE BOTH COHEN'S d VERSIONS FOR EACH PRIMARY OUTCOME
# ============================================================

rows = []
for outcome in ALL_PRIMARY:
    # Classical d on the diet-phase subset
    x_ctrl = diet_only[diet_only["group"] == "CTRL"][outcome].values
    x_hf   = diet_only[diet_only["group"] == "HF"][outcome].values
    d_raw, pooled_sd = classical_cohens_d(x_ctrl, x_hf)

    # Adjusted d from the 06a model row
    r = results[(results["tier"] == "primary") &
                (results["outcome"] == outcome) &
                (results["effect"] == "group[T.HF]")]
    if len(r) == 1:
        beta = float(r.iloc[0]["coef"])
        var_mouse = float(r.iloc[0]["group_var"])
        var_resid = float(r.iloc[0]["residual_var"])
        d_adj = adjusted_cohens_d(beta, var_mouse, var_resid)
        p_raw = float(r.iloc[0]["p_value"])
        p_fdr = float(r.iloc[0]["p_fdr"])
    else:
        beta = var_mouse = var_resid = np.nan
        d_adj = np.nan
        p_raw = p_fdr = np.nan

    rows.append({
        "outcome": outcome,
        "label": LABEL_MAP.get(outcome, outcome),
        "mean_CTRL": float(np.nanmean(x_ctrl)),
        "mean_HF":   float(np.nanmean(x_hf)),
        "n_CTRL": int(np.sum(~np.isnan(x_ctrl))),
        "n_HF":   int(np.sum(~np.isnan(x_hf))),
        "pooled_sd": pooled_sd,
        "cohens_d_classical": d_raw,
        "cohens_d_classical_label": cohen_label(d_raw),
        "cohens_d_adjusted": d_adj,
        "cohens_d_adjusted_label": cohen_label(d_adj),
        "beta_from_model": beta,
        "var_mouse": var_mouse,
        "var_resid": var_resid,
        "p_raw": p_raw,
        "p_fdr": p_fdr,
    })

eff = pd.DataFrame(rows)
eff.to_csv(OUT_CSV, index=False)
print(f"\nSaved effect-size table: {OUT_CSV}")


# ============================================================
# 5. CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("COHEN'S D — primary outcomes")
print("=" * 78)
print(f"{'outcome':<28s} {'d_class':>8s} {'d_adj':>8s} "
      f"{'|d_adj|':>8s} {'magnitude':>11s} {'p_fdr':>8s}")
for _, r in eff.sort_values("cohens_d_adjusted").iterrows():
    print(f"{r['outcome']:<28s} "
          f"{r['cohens_d_classical']:>+8.2f} "
          f"{r['cohens_d_adjusted']:>+8.2f} "
          f"{abs(r['cohens_d_adjusted']):>8.2f} "
          f"{r['cohens_d_adjusted_label']:>11s} "
          f"{r['p_fdr']:>8.3f}")


# ============================================================
# 6. FIGURE — COHEN'S D FOREST PLOT
# ------------------------------------------------------------
# One axis with two columns of bars per outcome: classical d on
# the left, adjusted d on the right. Vertical reference lines at
# the Cohen thresholds. Bars coloured red if the corresponding
# model p_fdr was significant.
# ============================================================

eff_sorted = eff.sort_values("cohens_d_adjusted").reset_index(drop=True)
n = len(eff_sorted)
y_pos = np.arange(n)

fig, ax = plt.subplots(figsize=(11, 7))

bar_h = 0.36

# Classical d
colours_c = [C_NS] * n   # classical d not directly tied to p_fdr
ax.barh(y_pos - bar_h / 2, eff_sorted["cohens_d_classical"],
        bar_h, color=colours_c, alpha=0.55, edgecolor="white",
        linewidth=0.5, label="Classical (raw data)")

# Adjusted d — colour by FDR significance
colours_a = [C_SIG if p < 0.05 else C_NS for p in eff_sorted["p_fdr"]]
ax.barh(y_pos + bar_h / 2, eff_sorted["cohens_d_adjusted"],
        bar_h, color=colours_a, alpha=0.90, edgecolor="white",
        linewidth=0.5, label="Adjusted (mixed model)")

# Cohen reference lines
for t, name in [(0.2, "small"), (0.5, "medium"),
                (0.8, "large"), (1.2, "very large")]:
    ax.axvline(t,  color="grey", linestyle=":", linewidth=0.6,
               alpha=0.55)
    ax.axvline(-t, color="grey", linestyle=":", linewidth=0.6,
               alpha=0.55)
ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)

# Annotate the magnitude label for each adjusted d
for i, r in eff_sorted.iterrows():
    d_adj = r["cohens_d_adjusted"]
    if np.isnan(d_adj):
        continue
    x_text = d_adj + (0.05 if d_adj >= 0 else -0.05)
    ha = "left" if d_adj >= 0 else "right"
    is_sig = r["p_fdr"] < 0.05
    star = "  *" if is_sig else ""
    ax.text(x_text, i + bar_h / 2,
            f"{r['cohens_d_adjusted_label']}{star}",
            va="center", ha=ha, fontsize=8.5,
            color=C_SIG if is_sig else "#444",
            fontweight="bold" if is_sig else "normal")

ax.set_yticks(y_pos)
ax.set_yticklabels(eff_sorted["label"], fontsize=10)
ax.set_xlabel("Cohen's d  (negative = HF lower; positive = HF higher)")
ax.set_title("Cohen's d effect sizes — primary outcomes\n"
             "(classical vs mixed-model-adjusted)",
             loc="left")

ax.legend(handles=[
    Patch(facecolor=C_NS,  alpha=0.55, label="Classical d"),
    Patch(facecolor=C_SIG, alpha=0.90, label="Adjusted d (FDR sig.)"),
    Patch(facecolor=C_NS,  alpha=0.90, label="Adjusted d (n.s.)"),
], frameon=False, fontsize=9, loc="lower right")

caption = ("Cohen's d for the 11 primary outcomes. Classical d uses "
           "raw HF vs CTRL group means and pooled SD; adjusted d "
           "standardises the mixed-model group coefficient by total "
           "model variance (mouse + residual). "
           "Reference lines: 0.2 small, 0.5 medium, 0.8 large, "
           "1.2 very large. Stars mark p_fdr < 0.05.")
fig.text(0.5, -0.04, caption, ha="center", fontsize=8,
         color="#444", wrap=True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved effect-size figure: {OUT_PNG}")


# ============================================================
# 7. SAVE HUMAN-READABLE TXT SUMMARY
# ------------------------------------------------------------
# Compact two-table summary that you can open in any editor and
# paste into emails or thesis appendices. Same numbers as the CSV.
# ============================================================

lines = []
lines.append("=" * 78)
lines.append(f"06E EFFECT SIZES (COHEN'S D) — {CABLE}")
lines.append("=" * 78)
lines.append("")
lines.append(f"Sample:  {len(diet_only)} recordings / "
             f"{diet_only['mouse'].nunique()} mice (diet-phase only)")
lines.append("")
lines.append("Cohen's d interpretation (Cohen 1988):")
lines.append("  |d| < 0.2   negligible")
lines.append("  0.2 - 0.5   small")
lines.append("  0.5 - 0.8   medium")
lines.append("  0.8 - 1.2   large")
lines.append("  > 1.2       very large")
lines.append("")
lines.append("Convention: positive d = HF higher; negative d = HF lower.")
lines.append("")
lines.append("-" * 78)
lines.append("CLASSICAL COHEN'S D (raw data, no covariate adjustment)")
lines.append("-" * 78)
lines.append(f"{'outcome':<30s} {'d':>8s} {'magnitude':>13s}")
for _, r in eff.sort_values("cohens_d_classical").iterrows():
    lines.append(f"{r['outcome']:<30s} "
                 f"{r['cohens_d_classical']:>+8.2f} "
                 f"{r['cohens_d_classical_label']:>13s}")
lines.append("")
lines.append("-" * 78)
lines.append("ADJUSTED COHEN'S D (mixed-model coefficient / total SD)")
lines.append("-" * 78)
lines.append(f"{'outcome':<30s} {'d_adj':>8s} {'magnitude':>13s} "
             f"{'p_fdr':>8s}")
for _, r in eff.sort_values("cohens_d_adjusted").iterrows():
    sig = "*" if r["p_fdr"] < 0.05 else ""
    lines.append(f"{r['outcome']:<30s} "
                 f"{r['cohens_d_adjusted']:>+8.2f} "
                 f"{r['cohens_d_adjusted_label']:>13s} "
                 f"{r['p_fdr']:>8.3f} {sig}")
lines.append("")
lines.append("Note: classical d is essentially zero for all band powers")
lines.append("      because the body-weight difference cancels much of")
lines.append("      the diet signal in the raw data. Adjusted d recovers")
lines.append("      the diet effect after controlling for body weight,")
lines.append("      days on diet, and estrous phase. This is the central")
lines.append("      methodological argument for using mixed-effects models.")
lines.append("")
lines.append("'*' marks rows where the mixed-model p_FDR < 0.05.")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary: {OUT_TXT}")


print("\nSTEP 06E finished successfully.")