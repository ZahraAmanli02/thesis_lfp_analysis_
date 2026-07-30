# ============================================================
# 16_RUN_MIXED_MODELS_POOLED.PY
#
# Purpose:
#   Re-run the primary linear mixed-effects analysis of Section
#   3.8 on the POOLED Cable 1 + Cable 3 dataset, so that the
#   inferential framework operates on the same 208-recording,
#   19-mouse feature matrix as the mouse-cluster bootstrap of
#   Sections 3.9 and script 10b_bootstrap_full.py.
#
# Model (per outcome y):
#   y_ij = β0 + β1·group + β2·phase + β3·weight
#        + β4·days_on_diet + β5·cable + u_i + ε_ij
#   u_i  ~ N(0, τ²)          random intercept per mouse
#   ε_ij ~ N(0, σ²)
#
#   cable ∈ {Cable 1, Cable 3} enters as a fixed-effect NUISANCE
#   covariate that absorbs any residual difference between the two
#   electrode-pair configurations after CFD construction. It is not
#   a scientific effect of interest; the reported β5 is documented
#   for transparency but not used for inference.
#
#   Reference categories:
#     group:  CTRL
#     phase:  A (pro-estrus)
#     cable:  Cable 1
#
# Feature families (same as 06a):
#   PRIMARY     — 6 log bands + 15 log ratios          (21 outcomes)
#   FDR         — Benjamini-Hochberg across the 21 primary p-values
#                 for the group[T.HF] main effect.
#
# Diet phase only:
#   As in 06a, baseline and recovery recordings are dropped so that
#   days_on_diet is defined for every row. Diet-phase HF and CTRL
#   contribute all their post-injection recordings from both cables.
#
# Inputs:
#   outputs/10a_features_Cable1/10a_features_Cable1.csv
#   /Users/amanlizahra/Desktop/For CABLE 3/thesis_lfp_analysis/
#     outputs/10a_features_Cable3/10a_features_Cable3.csv
#
# Outputs:
#   outputs/16_mixed_models_pooled/
#       16_mixed_model_results_pooled.csv   long format, every
#                                           fixed effect of every
#                                           outcome, one row each
#       16_mixed_model_summary_pooled.txt   readable summary tables
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CABLE1_CSV = os.path.join(
    OUTPUT_DIR, "10a_features_Cable1", "10a_features_Cable1.csv"
)
CABLE3_CSV = (
    "/Users/amanlizahra/Desktop/For CABLE 3/thesis_lfp_analysis/"
    "outputs/10a_features_Cable3/10a_features_Cable3.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, "16_mixed_models_pooled")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "16_mixed_model_results_pooled.csv")
OUT_TXT = os.path.join(OUT_DIR, "16_mixed_model_summary_pooled.txt")

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
RATIOS = [
    "theta_delta", "beta_delta", "low_gamma_delta",
    "high_gamma_delta", "fast_gamma_delta",
    "beta_theta", "low_gamma_theta", "high_gamma_theta",
    "fast_gamma_theta",
    "low_gamma_beta", "high_gamma_beta", "fast_gamma_beta",
    "high_gamma_low_gamma", "fast_gamma_low_gamma",
    "fast_gamma_high_gamma",
]
PRIMARY_OUTCOMES = (
    [f"log_{b}_abs" for b in BANDS] +
    [f"log_{r}"     for r in RATIOS]
)   # 6 + 15 = 21 primary outcomes


# ============================================================
# 2. LOAD & POOL
# ============================================================

for p in (CABLE1_CSV, CABLE3_CSV):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing 10a feature file:\n{p}")

c1 = pd.read_csv(CABLE1_CSV)
c3 = pd.read_csv(CABLE3_CSV)
pooled = pd.concat([c1, c3], ignore_index=True)

# diet-phase only (days_on_diet must be defined and > 0)
pooled = pooled[pooled["diet_phase"] == "diet"].copy()
pooled = pooled.dropna(subset=["days_on_diet", "body_weight",
                               "estrous_phase"])
pooled["mouse_uid"] = pooled["mouse"].astype(str)

print(f"Cable1: {int((pooled['cable']=='Cable1').sum()):>4d} rec")
print(f"Cable3: {int((pooled['cable']=='Cable3').sum()):>4d} rec")
print(f"Pooled: {len(pooled):>4d} rec  "
      f"({pooled['mouse_uid'].nunique()} unique mice; "
      f"{int((pooled['group']=='HF').sum())} HF / "
      f"{int((pooled['group']=='CTRL').sum())} CTRL)")


# ============================================================
# 3. FIT ONE OUTCOME
# ============================================================

FORMULA = ("{y} ~ C(group, Treatment(reference='CTRL'))"
           " + C(estrous_phase, Treatment(reference='A'))"
           " + body_weight + days_on_diet"
           " + C(cable, Treatment(reference='Cable1'))")


def fit_one(y_name, df):
    formula = FORMULA.format(y=y_name)
    model = smf.mixedlm(formula, df, groups=df["mouse_uid"])
    try:
        fit = model.fit(reml=True, method="lbfgs")
    except Exception as exc:
        return {"outcome": y_name, "converged": False,
                "reason": str(exc)}

    coefs = fit.fe_params.to_dict()
    ses = fit.bse_fe.to_dict()
    pvals = fit.pvalues.to_dict()
    cis = fit.conf_int().to_dict("index")

    rows = []
    for term in coefs:
        rows.append({
            "outcome": y_name,
            "term": term,
            "beta": coefs[term],
            "se": ses[term],
            "ci_lo": cis[term][0],
            "ci_hi": cis[term][1],
            "p_wald": pvals[term],
            "converged": fit.converged,
            "n_obs": int(fit.nobs),
            "n_mice": int(df["mouse_uid"].nunique()),
        })
    return rows


# ============================================================
# 4. FIT ALL 21 PRIMARY OUTCOMES
# ============================================================

all_rows = []
for outcome in PRIMARY_OUTCOMES:
    sub = pooled.dropna(subset=[outcome]).copy()
    if len(sub) < 20 or sub["mouse_uid"].nunique() < 8:
        print(f"[skip] {outcome:30s}  too few obs after dropna")
        continue
    result = fit_one(outcome, sub)
    if isinstance(result, dict) and not result.get("converged", True):
        print(f"[fail] {outcome:30s}  {result.get('reason','')}")
        continue
    all_rows.extend(result)
    beta_group = next(r["beta"] for r in result if "group" in r["term"])
    p_group = next(r["p_wald"] for r in result if "group" in r["term"])
    print(f"[ok]   {outcome:30s}  β_group={beta_group:+.3f}  "
          f"p={p_group:.3f}")

df_res = pd.DataFrame(all_rows)


# ============================================================
# 5. BH-FDR across 21 group[T.HF] p-values
# ============================================================

mask_group = df_res["term"].str.contains("group") & \
             df_res["term"].str.contains("HF")
group_rows = df_res[mask_group].copy()
if len(group_rows):
    _, p_bh, _, _ = multipletests(group_rows["p_wald"].values,
                                  method="fdr_bh")
    group_rows["p_bh"] = p_bh
    df_res = df_res.merge(
        group_rows[["outcome", "term", "p_bh"]],
        on=["outcome", "term"], how="left"
    )
else:
    df_res["p_bh"] = np.nan


# ============================================================
# 6. WRITE OUTPUTS
# ============================================================

df_res.to_csv(OUT_CSV, index=False)

with open(OUT_TXT, "w") as fh:
    fh.write("=" * 78 + "\n")
    fh.write("16 — Mixed-effects models on POOLED Cable 1 + Cable 3\n")
    fh.write("=" * 78 + "\n\n")
    fh.write(f"n_recordings = {len(pooled)}\n")
    fh.write(f"n_mice       = {pooled['mouse_uid'].nunique()}\n")
    fh.write(f"HF / CTRL    = {int((pooled['group']=='HF').sum())} / "
             f"{int((pooled['group']=='CTRL').sum())}\n")
    fh.write(f"Cable 1 rec  = {int((pooled['cable']=='Cable1').sum())}\n")
    fh.write(f"Cable 3 rec  = {int((pooled['cable']=='Cable3').sum())}\n\n")

    fh.write("Model:\n")
    fh.write("  y ~ group + estrous_phase + body_weight\n")
    fh.write("    + days_on_diet + cable + (1|mouse)\n\n")

    fh.write("Primary group[T.HF] effect, BH-adjusted across "
             f"{len(group_rows)} outcomes:\n")
    fh.write("-" * 78 + "\n")
    fh.write(f"{'outcome':30s} {'beta':>8s} {'SE':>7s} "
             f"{'95% CI':>22s} {'p_wald':>8s} {'p_BH':>8s} "
             f"{'sig':>4s}\n")
    fh.write("-" * 78 + "\n")
    for _, r in group_rows.sort_values("p_bh").iterrows():
        star = "***" if r["p_bh"] < 0.001 else \
               "**"  if r["p_bh"] < 0.01  else \
               "*"   if r["p_bh"] < 0.05  else ""
        fh.write(f"{r['outcome']:30s} "
                 f"{r['beta']:+8.3f} {r['se']:7.3f} "
                 f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                 f"{r['p_wald']:8.3f} {r['p_bh']:8.3f} {star:>4s}\n")

    # cable nuisance effect summary
    mask_cable = df_res["term"].str.contains("cable")
    cable_rows = df_res[mask_cable]
    fh.write("\n" + "=" * 78 + "\n")
    fh.write("Nuisance cable[T.Cable3] effect (documented, "
             "NOT inferential):\n")
    fh.write("-" * 78 + "\n")
    for _, r in cable_rows.iterrows():
        fh.write(f"{r['outcome']:30s} β={r['beta']:+.3f} "
                 f"p={r['p_wald']:.3f}\n")

print(f"\nWrote {OUT_CSV}")
print(f"Wrote {OUT_TXT}")
