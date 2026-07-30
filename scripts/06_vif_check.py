# ============================================================
# 06_VIF_CHECK.PY
# Purpose:
# Collinearity check for the predictors used in the 06a mixed-effects
# model:
#   outcome ~ group * days_on_diet + body_weight + C(estrous_phase)
#
# Why this exists:
# group, days_on_diet and body_weight all sit in the same model.
# HF mice gain weight as they stay on the diet, so days_on_diet and
# body_weight could be correlated enough to destabilise their
# coefficients (inflated standard errors, unstable estimates) even
# though the model itself is specified correctly. The Variance
# Inflation Factor (VIF) quantifies this directly:
#   VIF_j = 1 / (1 - R_j^2)
# where R_j^2 comes from regressing predictor j on all the other
# predictors. VIF < 5 is fine, 5-10 deserves a mention, > 10 is a
# real problem worth revisiting.
#
# Two VIF tables are reported:
#   1. Main-effects design (group, days_on_diet, body_weight,
#      C(estrous_phase)) — the meaningful check.
#   2. Full 06a design (adds the group:days_on_diet interaction) —
#      shown for completeness. Interaction terms are structurally
#      correlated with their main effects, so high VIF here is
#      expected and not a red flag by itself; table 1 is the one
#      that matters.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#
# Output:
#   outputs/06_vif_check_<CABLE>/
#       06_vif_check_main_effects_<CABLE>.csv
#       06_vif_check_full_model_<CABLE>.csv
#       06_vif_check_summary_<CABLE>.txt
#       06_vif_check_<CABLE>.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix
from statsmodels.stats.outliers_influence import variance_inflation_factor


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

OUT_DIR = os.path.join(OUTPUT_DIR, f"06_vif_check_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_MAIN_CSV = os.path.join(OUT_DIR, f"06_vif_check_main_effects_{CABLE}.csv")
OUT_FULL_CSV = os.path.join(OUT_DIR, f"06_vif_check_full_model_{CABLE}.csv")
OUT_TXT = os.path.join(OUT_DIR, f"06_vif_check_summary_{CABLE}.txt")
OUT_PNG = os.path.join(OUT_DIR, f"06_vif_check_{CABLE}.png")

# Same right-hand side as 06a
MAIN_FORMULA = "group + days_on_diet + body_weight + C(estrous_phase)"
FULL_FORMULA = "group * days_on_diet + body_weight + C(estrous_phase)"

VIF_WARN = 5.0
VIF_STRONG_WARN = 10.0

# Colours (consistent with the rest of the 06 series)
C_OK    = "#4C72B0"   # VIF < 5, fine
C_WATCH = "#DD8452"   # VIF in [5, 10), worth a mention
C_WARN  = "#C44E52"   # VIF >= 10, real concern

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def vif_colour(v):
    if v >= VIF_STRONG_WARN:
        return C_WARN
    if v >= VIF_WARN:
        return C_WATCH
    return C_OK


# ============================================================
# 2. LOAD + FILTER TO DIET PHASE (same subset 06a models)
# ============================================================

if not os.path.exists(BAND_POWERS_PATH):
    raise FileNotFoundError(f"Missing 05a output:\n{BAND_POWERS_PATH}")

df = pd.read_csv(BAND_POWERS_PATH)
diet_only = df[df["days_on_diet"].notna()].copy()

n_obs = len(diet_only)
n_mice = diet_only["mouse"].nunique()
print(f"\nVIF check on {n_obs} recordings / {n_mice} mice (diet-phase only)")


# ============================================================
# 3. VIF HELPER
# ------------------------------------------------------------
# Build the design matrix with patsy (same encoding statsmodels uses
# internally for smf.mixedlm), drop the Intercept column (VIF for an
# intercept is not meaningful and is always inflated by construction),
# then compute VIF per remaining column.
# ============================================================

def compute_vif(data, formula):
    # statsmodels' variance_inflation_factor expects the intercept to
    # stay IN the design matrix (it centers internally); dropping it
    # first breaks that centering and inflates every VIF artificially.
    # Compute on the full matrix, then drop the Intercept row from the
    # report — a VIF for the intercept itself is not meaningful.
    X = dmatrix(formula, data=data, return_type="dataframe")
    vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    out = pd.DataFrame({"term": X.columns, "VIF": vifs})
    out = out[out["term"] != "Intercept"]
    return out.sort_values("VIF", ascending=False).reset_index(drop=True)


def flag(v):
    if v >= VIF_STRONG_WARN:
        return "  <-- WARN (>=10)"
    if v >= VIF_WARN:
        return "  (watch, >=5)"
    return ""


# ============================================================
# 4. MAIN-EFFECTS VIF (the meaningful table)
# ============================================================

print("\n" + "=" * 60)
print("1. MAIN-EFFECTS VIF (group, days_on_diet, body_weight, estrous)")
print("=" * 60)

main_vif = compute_vif(diet_only, MAIN_FORMULA)
main_vif.to_csv(OUT_MAIN_CSV, index=False)
for _, r in main_vif.iterrows():
    print(f"  {r['term']:<25s} VIF = {r['VIF']:6.2f}{flag(r['VIF'])}")

n_warn = (main_vif["VIF"] >= VIF_WARN).sum()
n_strong = (main_vif["VIF"] >= VIF_STRONG_WARN).sum()
if n_strong > 0:
    print(f"\n  {n_strong} term(s) with VIF >= 10 — collinearity is a real "
          f"concern, revisit the model.")
elif n_warn > 0:
    print(f"\n  {n_warn} term(s) with VIF in [5, 10) — worth a mention in "
          f"the thesis, not disqualifying.")
else:
    print(f"\n  All main-effect VIFs < 5 — no collinearity concern.")


# ============================================================
# 5. FULL-MODEL VIF (with the group:days_on_diet interaction)
# ------------------------------------------------------------
# Shown for completeness only. Interaction terms are structurally
# correlated with their main effects, so elevated VIF here is
# expected and NOT by itself evidence of a problem — the main-
# effects table above is the one that drives the interpretation.
# ============================================================

print("\n" + "=" * 60)
print("2. FULL 06A MODEL VIF (includes group:days_on_diet interaction)")
print("=" * 60)
print("  (interaction terms inflate their own main effects' VIF by")
print("   construction — expected, not a red flag by itself)\n")

full_vif = compute_vif(diet_only, FULL_FORMULA)
full_vif.to_csv(OUT_FULL_CSV, index=False)
for _, r in full_vif.iterrows():
    print(f"  {r['term']:<30s} VIF = {r['VIF']:6.2f}{flag(r['VIF'])}")


# ============================================================
# 6. FIGURE — TWO-PANEL VIF OVERVIEW
# ------------------------------------------------------------
# Panel A: main-effects VIF (the table that matters).
# Panel B: full 06a-model VIF (interaction included, for reference).
# Horizontal bars, sorted, colour-coded by the same OK/watch/warn
# thresholds used in the console + txt output, with reference lines
# at VIF = 5 and VIF = 10.
# ============================================================

def plot_vif_panel(ax, vif_df, title):
    plot_df = vif_df.sort_values("VIF", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    colours = [vif_colour(v) for v in plot_df["VIF"]]

    ax.barh(y_pos, plot_df["VIF"], color=colours, alpha=0.9,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["term"], fontsize=9)
    ax.axvline(VIF_WARN, color=C_WATCH, linestyle="--", linewidth=1.0,
               alpha=0.8, label=f"VIF = {VIF_WARN:.0f} (watch)")
    ax.axvline(VIF_STRONG_WARN, color=C_WARN, linestyle="--", linewidth=1.0,
               alpha=0.8, label=f"VIF = {VIF_STRONG_WARN:.0f} (concern)")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title(title, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    for y, v in zip(y_pos, plot_df["VIF"]):
        ax.text(v + max(plot_df["VIF"]) * 0.01, y, f"{v:.2f}",
                va="center", fontsize=8, color="#333")


fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.5))

plot_vif_panel(axA, main_vif, "A. Main-effects model\n(the table that matters)")
plot_vif_panel(axB, full_vif, "B. Full 06a model\n(with group:days_on_diet interaction)")

fig.suptitle(f"Step 06 — collinearity check (VIF), {CABLE} "
             f"({n_obs} recordings / {n_mice} mice)",
             fontsize=13, fontweight="bold", y=1.02)
fig.text(0.5, -0.04,
         "VIF < 5: no concern.  5-10: worth a mention.  >= 10: revisit the "
         "model. Panel B's interaction-driven inflation is expected and "
         "does not, by itself, indicate a problem — read panel A.",
         ha="center", fontsize=8.5, color="#444444", style="italic",
         wrap=True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved figure: {OUT_PNG}")


# ============================================================
# 7. SAVE HUMAN-READABLE TXT SUMMARY
# ============================================================

lines = []
lines.append("=" * 70)
lines.append(f"06 VIF CHECK (COLLINEARITY) — {CABLE}")
lines.append("=" * 70)
lines.append("")
lines.append(f"Sample: {n_obs} recordings / {n_mice} mice (diet-phase only, "
             f"same subset as 06a)")
lines.append("")
lines.append("Rule of thumb: VIF < 5 OK, 5-10 mention it, >= 10 real problem.")
lines.append("")
lines.append("-" * 70)
lines.append("MAIN-EFFECTS MODEL (the table that matters)")
lines.append(f"  {MAIN_FORMULA}")
lines.append("-" * 70)
for _, r in main_vif.iterrows():
    lines.append(f"  {r['term']:<25s} VIF = {r['VIF']:6.2f}{flag(r['VIF'])}")
lines.append("")
lines.append("-" * 70)
lines.append("FULL 06A MODEL (with group:days_on_diet interaction)")
lines.append(f"  {FULL_FORMULA}")
lines.append("-" * 70)
lines.append("  Interaction terms structurally inflate their main effects'")
lines.append("  VIF; elevated values here are expected, not diagnostic on")
lines.append("  their own. Use the main-effects table above to judge")
lines.append("  collinearity.")
lines.append("")
for _, r in full_vif.iterrows():
    lines.append(f"  {r['term']:<30s} VIF = {r['VIF']:6.2f}{flag(r['VIF'])}")
lines.append("")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved summary: {OUT_TXT}")
print(f"Saved main-effects VIF CSV: {OUT_MAIN_CSV}")
print(f"Saved full-model VIF CSV: {OUT_FULL_CSV}")

print("\nSTEP 06 VIF CHECK finished successfully.")
