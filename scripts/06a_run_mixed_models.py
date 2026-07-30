# ============================================================
# 06A_RUN_MIXED_MODELS.PY
# Purpose:
# Run linear mixed-effects models on band-power, band-ratio, and
# oscillation-episode features to test whether diet (HF vs CTRL)
# affects hypothalamic LFP activity, while controlling for body
# weight, estrous phase, and time on diet — and accounting for the
# fact that multiple recordings come from each mouse.
#
# Model (per outcome):
#   outcome ~ group * days_on_diet + body_weight + C(estrous_phase)
#   groups = mouse              (random intercept (1|mouse))
#
# Why mixed-effects?
#   The 89 diet-phase recordings come from 19 mice (CTRL 9 / HF 10).
#   Recordings from the same mouse are not independent (same brain,
#   same electrode, same animal-level traits). Treating them as
#   independent would inflate degrees of freedom and false-positive
#   risk (pseudoreplication). A random intercept per mouse tells
#   the model: "each mouse has its own baseline level — account
#   for that correlation when estimating the diet effect."
#
# Two analysis tiers:
#   PRIMARY     (11 models)  — band powers (6) + band ratios (5)
#                              FDR (Benjamini-Hochberg) applied to
#                              the 11 group[T.HF] main-effect p-values
#   EXPLORATORY (12 models)  — oscillation episode features
#                              not corrected; reported as follow-up
#
# Estrous phase:
#   Phase A (pro-estrus, per lab's swab convention) is used as the
#   reference category (default alphabetical ordering). Every other
#   phase coefficient (phase_B/C/D) is therefore read as "vs
#   pro-estrus". Cell sizes for the reference category are
#   unbalanced (CTRL x A = 4, HF x A = 11); this affects the
#   precision of the estrous coefficients but not the primary
#   group effect.
#
# Sample note:
#   days_on_diet is undefined for baseline and recovery recordings,
#   so models drop those rows automatically. Effective n = 89.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05b_band_ratios_<CABLE>/05b_band_ratios_<CABLE>.csv
#   outputs/05c_oscillation_episodes_<CABLE>/
#       05c_features_combined_with_episodes_<CABLE>.csv
#
# Output:
#   outputs/06a_mixed_models_<CABLE>/
#       06a_mixed_model_results_<CABLE>.csv     (long format — every
#                                                fixed effect of every
#                                                model, one row each)
#       06a_mixed_model_summary_<CABLE>.txt     (human-readable summary)
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# Mute the long convergence/REML warnings — convergence status is
# captured manually below, so the terminal stays readable.
warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

# Resolve project root from this script's location (scripts/ -> root),
# so paths work regardless of the project folder name.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}", f"05a_band_powers_{CABLE}.csv"
)
BAND_RATIOS_PATH = os.path.join(
    OUTPUT_DIR, f"05b_band_ratios_{CABLE}", f"05b_band_ratios_{CABLE}.csv"
)
COMBINED_PATH = os.path.join(
    OUTPUT_DIR, f"05c_oscillation_episodes_{CABLE}",
    f"05c_features_combined_with_episodes_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"06a_mixed_models_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, f"06a_mixed_model_results_{CABLE}.csv")
OUT_TXT = os.path.join(OUT_DIR, f"06a_mixed_model_summary_{CABLE}.txt")


# ============================================================
# 2. OUTCOME LISTS + FORMULA
# ------------------------------------------------------------
# Primary outcomes: band powers (log-transformed here) + band ratios
# (already log-transformed in 05b output as log_*).
# Exploratory: 05c episode features (raw, no log).
# ============================================================

# Bands (must match 05a column scheme)
BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]

# Primary — log band powers (created below in section 3)
PRIMARY_LOG_POWERS = [f"log_{b}_abs" for b in BANDS]

# Primary — log band ratios (already in 05b output)
PRIMARY_LOG_RATIOS = [
    "log_theta_delta", "log_low_gamma_theta", "log_high_gamma_theta",
    "log_fast_gamma_theta", "log_beta_theta",
]

# Exploratory — episode features (3 bands x 4 metrics = 12)
EPISODE_BANDS = ["beta", "low_gamma", "high_gamma"]
EPISODE_METRICS = ["episode_rate", "mean_duration_sec",
                   "mean_amplitude", "fraction_of_time"]
EXPLORATORY_OUTCOMES = [f"{b}_{m}" for b in EPISODE_BANDS
                        for m in EPISODE_METRICS]

# Right-hand side of the formula — same for every outcome
RHS_FORMULA = ("group * days_on_diet + body_weight "
               "+ C(estrous_phase)")


# ============================================================
# 3. LOAD + BUILD MASTER TABLE
# ------------------------------------------------------------
# 05a is the base (band powers + metadata). 05b log_* columns and
# 05c episode-feature columns are merged onto it via differential_file
# (1:1 because all three scripts already use the same clean set).
# log_<band>_abs columns are added here (statistical transform).
# ============================================================

for p, step in [(BAND_POWERS_PATH, "05a"),
                (BAND_RATIOS_PATH, "05b"),
                (COMBINED_PATH, "05c")]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {step} output:\n{p}")

df_05a = pd.read_csv(BAND_POWERS_PATH)
df_05b = pd.read_csv(BAND_RATIOS_PATH)
df_05c = pd.read_csv(COMBINED_PATH)

# Add log band powers (statistical transform)
for band in BANDS:
    df_05a[f"log_{band}_abs"] = np.log10(df_05a[f"{band}_abs"])

# Merge log_* ratios from 05b
df = df_05a.merge(
    df_05b[["differential_file"] + PRIMARY_LOG_RATIOS],
    on="differential_file", how="left", validate="one_to_one"
)

# Merge episode features from 05c
df = df.merge(
    df_05c[["differential_file"] + EXPLORATORY_OUTCOMES],
    on="differential_file", how="left", validate="one_to_one"
)

print(f"\nMaster table: {df.shape[0]} recordings x {df.shape[1]} cols")


# ============================================================
# 4. FILTER TO DIET PHASE
# ------------------------------------------------------------
# Drop NaN-days_on_diet rows (baseline + recovery): these recordings
# cannot enter a model that uses days_on_diet as a predictor.
# Resulting set: 89 recordings / 19 mice.
# ============================================================

diet_only = df[df["days_on_diet"].notna()].copy()
n_obs = len(diet_only)
n_mice = diet_only["mouse"].nunique()
print(f"Diet-phase subset: {n_obs} recordings / {n_mice} mice")
print(f"  CTRL: {len(diet_only[diet_only['group']=='CTRL'])} rec / "
      f"{diet_only[diet_only['group']=='CTRL']['mouse'].nunique()} mice")
print(f"  HF:   {len(diet_only[diet_only['group']=='HF'])} rec / "
      f"{diet_only[diet_only['group']=='HF']['mouse'].nunique()} mice")


# ============================================================
# 5. MIXED-MODEL HELPER
# ------------------------------------------------------------
# Fit one model for one outcome variable, return a long-format
# DataFrame: one row per fixed-effect term (coef, SE, p, CI).
#
# Why long format?
#   - One row per (outcome, effect) makes filtering, FDR correction,
#     and plotting trivial with pandas (df.query, groupby, etc.).
#   - Wide format with 11 outcome columns x 8 effect rows is harder
#     to manipulate downstream.
# ============================================================

def fit_mixed_model(data, outcome):
    """
    Fit:  outcome ~ group * days_on_diet + body_weight + C(estrous_phase)
    Random intercept: (1|mouse)
    Return a long-format DataFrame with one row per fixed effect.
    """
    formula = f"{outcome} ~ {RHS_FORMULA}"

    try:
        model = smf.mixedlm(formula, data=data, groups=data["mouse"])
        # lbfgs is robust at these sample sizes; powell is the
        # fallback if lbfgs fails to converge.
        result = model.fit(method="lbfgs", reml=True)
        converged = bool(result.converged)
    except Exception:
        # Fallback method
        try:
            result = smf.mixedlm(formula, data=data,
                                 groups=data["mouse"]).fit(method="powell",
                                                           reml=True)
            converged = bool(result.converged)
        except Exception as e2:
            print(f"  FAIL ({outcome}): {e2}")
            return pd.DataFrame()

    # Extract fixed-effect terms (skip the 'Group Var' row, that is
    # the random-effect variance and is handled separately below).
    fixed_terms = [t for t in result.params.index if t != "Group Var"]

    rows = []
    for term in fixed_terms:
        coef = float(result.params[term])
        se = float(result.bse[term])
        pval = float(result.pvalues[term])
        try:
            ci_low, ci_high = result.conf_int().loc[term].tolist()
        except Exception:
            ci_low, ci_high = np.nan, np.nan
        rows.append({
            "outcome": outcome,
            "effect": term,
            "coef": coef,
            "se": se,
            "p_value": pval,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_obs": int(result.nobs),
            "n_groups": int(data["mouse"].nunique()),
            "group_var": float(result.cov_re.iloc[0, 0]),
            "residual_var": float(result.scale),
            "converged": converged,
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. RUN PRIMARY MODELS (11)
# ------------------------------------------------------------
# 6 log band powers + 5 log band ratios = 11 primary outcomes.
# These get FDR correction in section 8.
# ============================================================

print("\n" + "=" * 60)
print("PRIMARY MODELS (band powers + band ratios)")
print("=" * 60)

primary_rows = []
all_primary = PRIMARY_LOG_POWERS + PRIMARY_LOG_RATIOS

for outcome in all_primary:
    print(f"  fitting: {outcome}")
    r = fit_mixed_model(diet_only, outcome)
    if len(r) > 0:
        r["tier"] = "primary"
        primary_rows.append(r)
        # Report convergence + the main group effect inline
        grp_row = r[r["effect"] == "group[T.HF]"]
        if len(grp_row) == 1:
            conv = "OK" if grp_row.iloc[0]["converged"] else "WARN"
            print(f"    [{conv}] group[T.HF]: "
                  f"coef={grp_row.iloc[0]['coef']:+.3f}, "
                  f"p={grp_row.iloc[0]['p_value']:.3f}")

primary_df = pd.concat(primary_rows, ignore_index=True) \
    if primary_rows else pd.DataFrame()


# ============================================================
# 7. RUN EXPLORATORY MODELS (12)
# ------------------------------------------------------------
# Episode features. No FDR correction. Reported as follow-up.
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY MODELS (oscillation episode features)")
print("=" * 60)

exploratory_rows = []
for outcome in EXPLORATORY_OUTCOMES:
    print(f"  fitting: {outcome}")
    r = fit_mixed_model(diet_only, outcome)
    if len(r) > 0:
        r["tier"] = "exploratory"
        exploratory_rows.append(r)
        grp_row = r[r["effect"] == "group[T.HF]"]
        if len(grp_row) == 1:
            conv = "OK" if grp_row.iloc[0]["converged"] else "WARN"
            print(f"    [{conv}] group[T.HF]: "
                  f"coef={grp_row.iloc[0]['coef']:+.3f}, "
                  f"p={grp_row.iloc[0]['p_value']:.3f}")

exploratory_df = pd.concat(exploratory_rows, ignore_index=True) \
    if exploratory_rows else pd.DataFrame()


# ============================================================
# 8. FDR CORRECTION
# ------------------------------------------------------------
# Benjamini-Hochberg on the 11 primary group[T.HF] p-values.
# Exploratory tier is left untouched.
#
# The FDR-adjusted p is added as a new column p_fdr that is filled
# only for the primary group main-effect rows; all other rows have
# p_fdr = NaN.
# ============================================================

all_results = pd.concat([primary_df, exploratory_df], ignore_index=True)
all_results["p_fdr"] = np.nan

if len(primary_df) > 0:
    primary_group_mask = ((all_results["tier"] == "primary") &
                          (all_results["effect"] == "group[T.HF]"))
    pvals = all_results.loc[primary_group_mask, "p_value"].values

    if len(pvals) > 0:
        _, p_fdr, _, _ = multipletests(pvals, method="fdr_bh")
        all_results.loc[primary_group_mask, "p_fdr"] = p_fdr
        print(f"\nFDR (Benjamini-Hochberg) applied to "
              f"{len(pvals)} primary group[T.HF] p-values")


# ============================================================
# 9. SAVE — LONG-FORMAT CSV
# ============================================================

# Column order: identification first, then statistics, then bookkeeping
col_order = ["tier", "outcome", "effect",
             "coef", "se", "ci_low", "ci_high",
             "p_value", "p_fdr",
             "n_obs", "n_groups",
             "group_var", "residual_var", "converged"]
all_results = all_results[col_order]
all_results.to_csv(OUT_CSV, index=False)
print(f"\nSaved results CSV:\n{OUT_CSV}")
print(f"  {len(all_results)} rows "
      f"({all_results['outcome'].nunique()} outcomes)")


# ============================================================
# 10. SAVE — HUMAN-READABLE TXT
# ------------------------------------------------------------
# Two compact tables: one for primary group[T.HF] effects (with FDR),
# one for exploratory group[T.HF] effects. Everything else stays in
# the long-format CSV for downstream analysis / plotting.
# ============================================================

lines = []
lines.append("=" * 70)
lines.append(f"06A MIXED-MODEL SUMMARY — {CABLE}")
lines.append("=" * 70)
lines.append("")
lines.append(f"Model:   outcome ~ {RHS_FORMULA}")
lines.append(f"Random:  (1 | mouse)")
lines.append(f"Method:  REML (statsmodels.MixedLM, lbfgs)")
lines.append(f"Sample:  {n_obs} recordings / {n_mice} mice "
             f"(diet-phase only)")
lines.append(f"Estrous: A is reference (Caligioni 2009)")
lines.append("")
lines.append("-" * 70)
lines.append("PRIMARY — group[T.HF] main effect (with FDR-BH correction)")
lines.append("-" * 70)
lines.append(f"{'outcome':<30s} {'coef':>8s} {'se':>7s}"
             f" {'p_raw':>8s} {'p_fdr':>8s} {'conv':>5s}")

primary_grp = all_results[
    (all_results["tier"] == "primary") &
    (all_results["effect"] == "group[T.HF]")
].sort_values("p_value")
for _, r in primary_grp.iterrows():
    sig = "*" if r["p_fdr"] < 0.05 else ""
    lines.append(f"{r['outcome']:<30s} {r['coef']:>+8.3f} {r['se']:>7.3f}"
                 f" {r['p_value']:>8.3f} {r['p_fdr']:>8.3f}"
                 f" {'Yes' if r['converged'] else 'No':>5s} {sig}")

lines.append("")
lines.append("-" * 70)
lines.append("EXPLORATORY — group[T.HF] main effect (no FDR correction)")
lines.append("-" * 70)
lines.append(f"{'outcome':<30s} {'coef':>8s} {'se':>7s}"
             f" {'p_raw':>8s} {'conv':>5s}")

exp_grp = all_results[
    (all_results["tier"] == "exploratory") &
    (all_results["effect"] == "group[T.HF]")
].sort_values("p_value")
for _, r in exp_grp.iterrows():
    sig = "*" if r["p_value"] < 0.05 else ""
    lines.append(f"{r['outcome']:<30s} {r['coef']:>+8.3f} {r['se']:>7.3f}"
                 f" {r['p_value']:>8.3f}"
                 f" {'Yes' if r['converged'] else 'No':>5s} {sig}")

lines.append("")
lines.append("Note: '*' marks rows where the relevant p < 0.05.")
lines.append("      Full coefficients (days_on_diet, interaction,")
lines.append("      body_weight, estrous phases) are in the CSV.")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved readable summary:\n{OUT_TXT}")


# ============================================================
# 11. QUICK CONSOLE OUTPUT (top of summary)
# ============================================================

print("\n" + "\n".join(lines))

# Convergence check across all models
n_failed = (~all_results["converged"]).sum()
if n_failed > 0:
    print(f"\nWARN: {n_failed} effect rows from non-converged models")

print("\nSTEP 06A finished successfully.")