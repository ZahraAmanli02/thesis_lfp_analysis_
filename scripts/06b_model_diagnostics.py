# ============================================================
# 06B_MODEL_DIAGNOSTICS.PY
# Purpose:
# Diagnostic checks for the 11 primary mixed-effects models from
# 06a, to verify that the model assumptions hold and that the
# reported coefficients are trustworthy. Outputs are designed for
# the thesis appendix and for sharing with the supervisor.
#
# Why this matters:
#   Reviewer's first question on any mixed model is "did you check
#   the residuals?". A clean diagnostic set up front prevents that
#   question becoming a problem at viva or revision time. We check
#   the four standard assumptions of a linear mixed model:
#     1. Linearity        (residual vs fitted should be flat)
#     2. Normality        (QQ plot should follow the line)
#     3. Homoscedasticity (scale-location should be flat)
#     4. Random-effect    (per-mouse random intercepts should be
#        normality          roughly normally distributed)
#
# For each of the 11 primary outcomes we produce:
#   - a 4-panel diagnostic figure
#       Panel 1: residuals vs fitted        (linearity + homoscedasticity)
#       Panel 2: QQ plot of residuals       (normality)
#       Panel 3: scale-location              (homoscedasticity, log scale)
#       Panel 4: random intercepts per mouse (RE normality, outlier mice)
#
# And one overview figure:
#   - panel A: residual skewness across the 11 models
#   - panel B: residual std vs fitted range (variance health)
#   - panel C: Shapiro-Wilk p of residuals across the 11 models
#   - panel D: ICC per model (how much variance comes from mouse)
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05b_band_ratios_<CABLE>/05b_band_ratios_<CABLE>.csv
#   outputs/05c_oscillation_episodes_<CABLE>/
#       05c_features_combined_with_episodes_<CABLE>.csv
#
# Output:
#   outputs/06b_diagnostics_<CABLE>/
#       06b_diagnostics_<outcome>_<CABLE>.png   (11 per-model figures)
#       06b_diagnostics_overview_<CABLE>.png    (summary across models)
#       06b_diagnostics_summary_<CABLE>.csv     (numerical summary)
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

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
COMBINED_PATH = os.path.join(
    OUTPUT_DIR, f"05c_oscillation_episodes_{CABLE}",
    f"05c_features_combined_with_episodes_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"06b_diagnostics_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY_PATH = os.path.join(OUT_DIR, f"06b_diagnostics_summary_{CABLE}.csv")
OVERVIEW_PATH = os.path.join(
    OUT_DIR, f"06b_diagnostics_overview_{CABLE}.png"
)


# Bands + formula (must match 06a)
BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
PRIMARY_LOG_POWERS = [f"log_{b}_abs" for b in BANDS]
PRIMARY_LOG_RATIOS = [
    "log_theta_delta", "log_low_gamma_theta", "log_high_gamma_theta",
    "log_fast_gamma_theta", "log_beta_theta",
]
PRIMARY_OUTCOMES = PRIMARY_LOG_POWERS + PRIMARY_LOG_RATIOS

RHS_FORMULA = ("group * days_on_diet + body_weight "
               "+ C(estrous_phase)")

# Display labels
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
C_OK   = "#4C72B0"   # blue — assumption looks OK
C_WARN = "#C44E52"   # red — flag for attention
C_NEUT = "#888888"   # grey — neutral

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
})


# ============================================================
# 2. LOAD + BUILD MASTER, FILTER TO DIET PHASE
# ============================================================

df_05a = pd.read_csv(BAND_POWERS_PATH)
df_05b = pd.read_csv(BAND_RATIOS_PATH)
df_05c = pd.read_csv(COMBINED_PATH)

for band in BANDS:
    df_05a[f"log_{band}_abs"] = np.log10(df_05a[f"{band}_abs"])

df = df_05a.merge(
    df_05b[["differential_file"] + PRIMARY_LOG_RATIOS],
    on="differential_file", how="left", validate="one_to_one"
)

diet_only = df[df["days_on_diet"].notna()].copy()
n_obs = len(diet_only)
n_mice = diet_only["mouse"].nunique()
print(f"\nDiagnostics on {n_obs} recordings / {n_mice} mice")


# ============================================================
# 3. PER-MODEL DIAGNOSTIC HELPER
# ------------------------------------------------------------
# Fit the model, extract residuals + fitted + random intercepts,
# draw a 4-panel diagnostic figure, return the numeric summary.
# ============================================================

def diagnose_model(data, outcome):
    """
    Fit one model and produce a 4-panel diagnostic figure.
    Return a dict with summary numbers (for the overview figure).
    """
    formula = f"{outcome} ~ {RHS_FORMULA}"
    result = None
    for method in ["lbfgs", "powell", "bfgs"]:
        try:
            model = smf.mixedlm(formula, data=data, groups=data["mouse"])
            result = model.fit(method=method, reml=True)
            # Test that fitted values and random_effects are accessible
            _ = np.asarray(result.fittedvalues)
            _ = result.random_effects
            break
        except Exception:
            result = None
            continue

    if result is None:
        print(f"  FAIL ({outcome}): no fitting method converged cleanly")
        return None

    fitted = np.asarray(result.fittedvalues)
    residuals = np.asarray(result.resid)
    random_effects = result.random_effects  # dict: mouse -> Series

    # Random intercepts per mouse (the "Group" key holds the
    # intercept estimate in a random-intercept model)
    re_intercepts = pd.Series({
        m: re_dict.get("Group", np.nan)
        for m, re_dict in random_effects.items()
    }).dropna()

    # Diagnostic numbers
    resid_skew = float(stats.skew(residuals))
    # Shapiro-Wilk normality test of residuals
    sw_stat, sw_p = stats.shapiro(residuals)
    resid_std = float(np.std(residuals))
    fit_range = float(np.ptp(fitted))   # peak-to-peak of fitted

    # ICC = sigma_mouse / (sigma_mouse + sigma_resid)
    var_mouse = float(result.cov_re.iloc[0, 0])
    var_resid = float(result.scale)
    icc = var_mouse / (var_mouse + var_resid) \
        if (var_mouse + var_resid) > 0 else np.nan

    # ---- 4-panel diagnostic figure ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    label = LABEL_MAP.get(outcome, outcome)

    # Panel 1 — residuals vs fitted (linearity + homoscedasticity)
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.55, color=C_OK,
               edgecolors="white", linewidths=0.5, s=35)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    # Lowess trend (light overlay) — uses np.polyfit for simplicity
    try:
        z = np.polyfit(fitted, residuals, 2)
        xs = np.linspace(fitted.min(), fitted.max(), 100)
        ax.plot(xs, np.polyval(z, xs), color=C_WARN, linewidth=1.5,
                alpha=0.7, label="quadratic trend")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    except Exception:
        pass
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("1. Residuals vs Fitted\n"
                 "(want: flat scatter around 0)", loc="left")

    # Panel 2 — QQ plot of residuals (normality)
    ax = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    # probplot() sets its own centered title ("Probability Plot"); clear
    # it before setting the custom left-aligned one below, otherwise
    # matplotlib keeps both titles and they overlap on the render.
    ax.set_title("")
    # Recolour matplotlib defaults to match style
    ax.get_lines()[0].set_marker("o")
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_markerfacecolor(C_OK)
    ax.get_lines()[0].set_markeredgecolor("white")
    ax.get_lines()[0].set_alpha(0.7)
    ax.get_lines()[1].set_color(C_WARN)
    ax.get_lines()[1].set_linewidth(1.5)
    ax.set_title(f"2. QQ plot of residuals\n"
                 f"(Shapiro-Wilk p = {sw_p:.3f}; "
                 f"want: dots on the line)", loc="left")

    # Panel 3 — scale-location (homoscedasticity)
    ax = axes[1, 0]
    std_resid = (residuals - np.mean(residuals)) / np.std(residuals)
    sqrt_abs = np.sqrt(np.abs(std_resid))
    ax.scatter(fitted, sqrt_abs, alpha=0.55, color=C_OK,
               edgecolors="white", linewidths=0.5, s=35)
    try:
        z = np.polyfit(fitted, sqrt_abs, 1)
        xs = np.linspace(fitted.min(), fitted.max(), 100)
        ax.plot(xs, np.polyval(z, xs), color=C_WARN, linewidth=1.5,
                alpha=0.7, label="linear trend")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    except Exception:
        pass
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("sqrt(|standardised residuals|)")
    ax.set_title("3. Scale-location\n"
                 "(want: flat — equal variance)", loc="left")

    # Panel 4 — random intercepts per mouse (RE normality)
    ax = axes[1, 1]
    sorted_re = re_intercepts.sort_values()
    colours = [C_WARN if abs(v) > 2 * sorted_re.std() else C_OK
               for v in sorted_re.values]
    y_pos = np.arange(len(sorted_re))
    ax.barh(y_pos, sorted_re.values, color=colours, alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"M{int(m)}" for m in sorted_re.index],
                       fontsize=8)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Random intercept (deviation from group mean)")
    ax.set_title(f"4. Random intercepts per mouse\n"
                 f"(ICC = {icc:.2f}; red = > 2 SD outlier)", loc="left")

    fig.suptitle(f"Model diagnostics — {label}  ({outcome})",
                 fontsize=12, fontweight="bold", y=1.00)

    # Footer with key numbers
    footer = (f"residual skew = {resid_skew:+.2f}   |   "
              f"residual std = {resid_std:.3f}   |   "
              f"Shapiro p = {sw_p:.3f}   |   "
              f"ICC = {icc:.2f}   |   n = {len(residuals)} / "
              f"{len(re_intercepts)} mice")
    fig.text(0.5, -0.01, footer, ha="center", fontsize=9,
             color="#444", style="italic")

    plt.tight_layout()
    out_path = os.path.join(
        OUT_DIR, f"06b_diagnostics_{outcome}_{CABLE}.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "outcome": outcome,
        "label": label,
        "n_obs": int(len(residuals)),
        "n_mice": int(len(re_intercepts)),
        "resid_skew": resid_skew,
        "resid_std": resid_std,
        "fitted_range": fit_range,
        "shapiro_p": float(sw_p),
        "var_mouse": var_mouse,
        "var_resid": var_resid,
        "ICC": icc,
        "n_re_outliers": int(sum(
            abs(v) > 2 * sorted_re.std() for v in sorted_re.values
        )),
    }


# ============================================================
# 4. RUN DIAGNOSTICS FOR ALL 11 PRIMARY MODELS
# ============================================================

print("\n" + "=" * 60)
print("PRIMARY MODEL DIAGNOSTICS")
print("=" * 60)

summaries = []
for outcome in PRIMARY_OUTCOMES:
    print(f"  diagnosing: {outcome}")
    s = diagnose_model(diet_only, outcome)
    if s is not None:
        summaries.append(s)
        flag = ""
        if abs(s["resid_skew"]) > 1:
            flag += "  [skew]"
        if s["shapiro_p"] < 0.05:
            flag += "  [Shapiro p<0.05]"
        if s["n_re_outliers"] > 0:
            flag += f"  [{s['n_re_outliers']} RE outlier]"
        print(f"    skew={s['resid_skew']:+.2f}  "
              f"shapiro_p={s['shapiro_p']:.3f}  "
              f"ICC={s['ICC']:.2f}{flag}")

summary_df = pd.DataFrame(summaries)
summary_df.to_csv(SUMMARY_PATH, index=False)
print(f"\nSaved numeric summary: {SUMMARY_PATH}")


# ============================================================
# 5. OVERVIEW FIGURE (across all 11 models)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
n = len(summary_df)
y_pos = np.arange(n)
labels = summary_df["label"].values

# Panel A — residual skewness
ax = axes[0, 0]
colours = [C_WARN if abs(v) > 1 else C_OK
           for v in summary_df["resid_skew"]]
ax.barh(y_pos, summary_df["resid_skew"], color=colours, alpha=0.8,
        edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
ax.axvline(1, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)
ax.axvline(-1, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Residual skewness")
ax.set_title("A. Residual skewness across models\n"
             "(want: within +/-1)", loc="left")

# Panel B — Shapiro-Wilk p values
ax = axes[0, 1]
colours = [C_WARN if v < 0.05 else C_OK
           for v in summary_df["shapiro_p"]]
ax.barh(y_pos, summary_df["shapiro_p"], color=colours, alpha=0.8,
        edgecolor="white", linewidth=0.5)
ax.axvline(0.05, color=C_WARN, linewidth=0.8, linestyle="--",
           alpha=0.7, label="p = 0.05")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Shapiro-Wilk p (residual normality)")
ax.set_title("B. Residual normality test\n"
             "(want: p > 0.05)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")

# Panel C — ICC across models
ax = axes[1, 0]
ax.barh(y_pos, summary_df["ICC"], color=C_NEUT, alpha=0.8,
        edgecolor="white", linewidth=0.5)
ax.axvline(0.1, color="grey", linewidth=0.5, linestyle=":", alpha=0.5,
           label="0.1 (mixed model worth it)")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Intraclass correlation coefficient (ICC)")
ax.set_xlim(0, 1)
ax.set_title("C. Mouse-level clustering (ICC)\n"
             "(higher = mixed model more important)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")

# Panel D — residual std vs fitted range (variance health)
ax = axes[1, 1]
ax.scatter(summary_df["fitted_range"], summary_df["resid_std"],
           color=C_OK, s=60, alpha=0.8, edgecolors="white", linewidths=1)
for _, row in summary_df.iterrows():
    ax.annotate(row["label"].split(" ")[0],
                (row["fitted_range"], row["resid_std"]),
                fontsize=7, alpha=0.75,
                xytext=(4, 2), textcoords="offset points")
ax.set_xlabel("Fitted range")
ax.set_ylabel("Residual std")
ax.set_title("D. Variance health\n"
             "(want: residual std small vs fitted range)", loc="left")

fig.suptitle(f"Step 06 model diagnostics overview — {CABLE} "
             f"({n} primary models)",
             fontsize=13, fontweight="bold", y=1.00)

caption = (f"Diagnostic summary across the 11 primary mixed-effects "
           f"models (n={n_obs} recordings / {n_mice} mice).  "
           f"Red bars flag potential concerns: |skew| > 1, "
           f"Shapiro-Wilk p < 0.05.  ICC > 0.1 confirms that random "
           f"intercepts (mixed model) are warranted.")
fig.text(0.5, -0.01, caption, ha="center", fontsize=8, color="#444",
         style="italic")

plt.tight_layout()
plt.savefig(OVERVIEW_PATH, dpi=200, bbox_inches="tight",
            facecolor="white")
print(f"Saved overview figure: {OVERVIEW_PATH}")


# ============================================================
# 6. CONSOLE FLAGGING
# ============================================================

print("\n" + "=" * 60)
print("DIAGNOSTIC FLAGS")
print("=" * 60)

skew_flag = summary_df[summary_df["resid_skew"].abs() > 1]
sw_flag = summary_df[summary_df["shapiro_p"] < 0.05]
re_flag = summary_df[summary_df["n_re_outliers"] > 0]

if len(skew_flag) == 0 and len(sw_flag) == 0 and len(re_flag) == 0:
    print("All 11 models pass the standard diagnostic checks.")
else:
    if len(skew_flag) > 0:
        print(f"\nResidual skew |skew| > 1:")
        for _, r in skew_flag.iterrows():
            print(f"  {r['outcome']:30s} skew = {r['resid_skew']:+.2f}")
    if len(sw_flag) > 0:
        print(f"\nShapiro-Wilk p < 0.05 (residuals not strictly normal):")
        for _, r in sw_flag.iterrows():
            print(f"  {r['outcome']:30s} p = {r['shapiro_p']:.3f}")
    if len(re_flag) > 0:
        print(f"\nRandom-effect outliers (>2 SD):")
        for _, r in re_flag.iterrows():
            print(f"  {r['outcome']:30s} n_outliers = {r['n_re_outliers']}")
    print("\nNote: minor flags are common and not necessarily")
    print("invalidating. Per-model 4-panel figures show the detail.")

print("\nSTEP 06B finished successfully.")