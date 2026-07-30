# ============================================================
# 07_BODY_WEIGHT_ANALYSIS.PY
# Purpose:
# Generate Section 4.2 (Body weight trajectory) outputs:
#   Figure 4.4 — Body weight trajectory of all 19 mice over
#                the whole experimental period (cable-agnostic;
#                built from raw weighing sheet).
#   Table 4.1 — Mixed-effects model output per cable, using
#                recording-day body weights from the 05a table.
#
# Model:  body_weight ~ group * days_on_diet + (1|mouse)
#         fit on the diet-phase subset (matches 06a subset)
#
# Runs one cable at a time (set CABLE below). The trajectory
# figure is cable-agnostic — both runs produce the same file
# (harmless overwrite). The model CSV is per-cable.
#
# Input:
#   data/Overview_Weighing_CD1.xlsx      (Sheet: Tabelle1)
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#
# Output:
#   outputs/07_body_weight_analysis/
#       Figure_4_4_body_weight_trajectory.{png,pdf}
#       07_body_weight_model_<CABLE>.csv
#       07_body_weight_baseline_check.csv       (cable-agnostic)
# ============================================================

import os
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

WEIGHING_PATH = os.path.join(BASE_DIR, "data", "Overview_Weighing_CD1.xlsx")
BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}", f"05a_band_powers_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, "07_body_weight_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# Experimental anchors
DIET_START = pd.Timestamp("2026-02-23")     # weighing morning; diet introduced ~1.30pm
RECOVERY_START = pd.Timestamp("2026-03-30") # old food re-introduced

# HF mouse IDs (from Excel note "HF: 10x")
HF_MICE = {1, 4, 9, 10, 14, 17, 21, 22, 23, 25}

# Colours (match 06_pre_data_check.py)
C_CTRL = "#4C72B0"
C_HF   = "#C44E52"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. LOAD + RESHAPE RAW WEIGHING SHEET
# ------------------------------------------------------------
# Long format: one row per (mouse, date) pair, with body weight.
# Group assigned from the HF_MICE set. Day zero = 2026-02-23
# (baseline weighing done before diet introduction that afternoon).
# ============================================================

if not os.path.exists(WEIGHING_PATH):
    raise FileNotFoundError(f"Missing weighing sheet:\n{WEIGHING_PATH}")

df_wide = pd.read_excel(WEIGHING_PATH, sheet_name="Tabelle1", header=2)
# Keep only rows whose first column is numeric (i.e. a mouse ID)
mask = pd.to_numeric(df_wide.iloc[:, 0], errors="coerce").notna()
df_wide = df_wide[mask].copy()
df_wide = df_wide.rename(columns={df_wide.columns[0]: "mouse"})
df_wide["mouse"] = df_wide["mouse"].astype(int)
df_wide["group"] = df_wide["mouse"].apply(lambda m: "HF" if m in HF_MICE else "CTRL")

# All date columns are datetime.datetime / pd.Timestamp
date_cols = [c for c in df_wide.columns
             if isinstance(c, (pd.Timestamp, datetime.datetime))]

df_long = df_wide.melt(
    id_vars=["mouse", "group"], value_vars=date_cols,
    var_name="date", value_name="body_weight"
)
df_long["date"] = pd.to_datetime(df_long["date"])
df_long["day"] = (df_long["date"] - DIET_START).dt.days
df_long = df_long.dropna(subset=["body_weight"])

print(f"Raw weighings: {len(df_long)} across "
      f"{df_long['day'].nunique()} dates for "
      f"{df_long['mouse'].nunique()} mice "
      f"(CTRL {df_long[df_long['group']=='CTRL']['mouse'].nunique()}, "
      f"HF {df_long[df_long['group']=='HF']['mouse'].nunique()})")


# ============================================================
# 3. BASELINE EQUIVALENCE CHECK (cable-agnostic)
# ------------------------------------------------------------
# Confirms the two groups were matched at day 0 (before the diet
# was introduced). Written once, on the first run only.
# ============================================================

baseline_path = os.path.join(OUT_DIR, "07_body_weight_baseline_check.csv")

baseline = df_long[df_long["day"] == 0]
ctrl0 = baseline[baseline["group"] == "CTRL"]["body_weight"].values
hf0   = baseline[baseline["group"] == "HF"]["body_weight"].values
t_stat, p_val = stats.ttest_ind(ctrl0, hf0, equal_var=False)

baseline_summary = pd.DataFrame({
    "group": ["CTRL", "HF"],
    "n":     [len(ctrl0), len(hf0)],
    "mean":  [ctrl0.mean(), hf0.mean()],
    "std":   [ctrl0.std(ddof=1), hf0.std(ddof=1)],
    "sem":   [ctrl0.std(ddof=1) / np.sqrt(len(ctrl0)),
              hf0.std(ddof=1) / np.sqrt(len(hf0))],
})
baseline_summary["welch_t"] = t_stat
baseline_summary["welch_p"] = p_val
baseline_summary.to_csv(baseline_path, index=False)
print(f"\nBaseline (day 0):")
print(f"  CTRL {ctrl0.mean():.2f} ± {ctrl0.std(ddof=1)/np.sqrt(len(ctrl0)):.2f}g "
      f"vs HF {hf0.mean():.2f} ± {hf0.std(ddof=1)/np.sqrt(len(hf0)):.2f}g   "
      f"(Welch t = {t_stat:.3f}, p = {p_val:.4f})")


# ============================================================
# 4. FIGURE 4.4 — Trajectory (cable-agnostic)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

# Individual mouse traces
for grp, col in [("CTRL", C_CTRL), ("HF", C_HF)]:
    sub = df_long[df_long["group"] == grp]
    for mouse in sub["mouse"].unique():
        m = sub[sub["mouse"] == mouse].sort_values("day")
        ax.plot(m["day"], m["body_weight"], color=col, alpha=0.25, lw=0.8)

# Group means + SEM
for grp, col in [("CTRL", C_CTRL), ("HF", C_HF)]:
    s = df_long[df_long["group"] == grp].groupby("day")["body_weight"] \
        .agg(["mean", "sem"]).reset_index()
    ax.plot(s["day"], s["mean"], color=col, lw=2.5,
            label=f"{grp} (mean ± SEM)")
    ax.fill_between(s["day"], s["mean"] - s["sem"], s["mean"] + s["sem"],
                    color=col, alpha=0.25)

# Diet-phase shading
recovery_day = (RECOVERY_START - DIET_START).days
ax.axvspan(0, recovery_day, alpha=0.05, color="orange", zorder=0)

# Recovery start marker
ax.axvline(recovery_day, color="grey", ls="--", lw=1, alpha=0.7)
ax.text(recovery_day + 0.5, ax.get_ylim()[1] * 0.98,
        "Recovery start\n(Mar 30)", ha="left", va="top",
        fontsize=8, color="grey")

ax.set_xlabel("Days from diet introduction (Feb 23, 2026)")
ax.set_ylabel("Body weight (g)")
ax.set_title(f"Body weight trajectory "
             f"(n = {df_long['mouse'].nunique()} mice, "
             f"{df_long['day'].nunique()} weighings)", fontsize=11)
ax.legend(loc="lower left", frameon=False, fontsize=9)
ax.set_xlim(-1, df_long["day"].max() + 1)

plt.tight_layout()
fig_png = os.path.join(OUT_DIR, "Figure_4_4_body_weight_trajectory.png")
fig_pdf = os.path.join(OUT_DIR, "Figure_4_4_body_weight_trajectory.pdf")
plt.savefig(fig_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(fig_pdf,           bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {fig_png}")


# ============================================================
# 5. MIXED-EFFECTS MODEL (per cable, on 05a diet-phase subset)
# ------------------------------------------------------------
# Uses recording-day body weights from 05a (measured or
# interpolated per Section 3.3.2). Restricted to diet phase,
# where days_on_diet is defined and where the models of Step 06
# operate.
# ============================================================

if not os.path.exists(BAND_POWERS_PATH):
    raise FileNotFoundError(f"Missing 05a output for {CABLE}:\n{BAND_POWERS_PATH}")

df = pd.read_csv(BAND_POWERS_PATH)
diet = df[df["diet_phase"] == "diet"].copy()
diet["group"] = pd.Categorical(diet["group"], categories=["CTRL", "HF"])

n_rec = len(diet)
n_mice_ctrl = diet[diet["group"] == "CTRL"]["mouse"].nunique()
n_mice_hf   = diet[diet["group"] == "HF"]["mouse"].nunique()
print(f"\nMixed model input ({CABLE}): {n_rec} recordings "
      f"(CTRL {n_mice_ctrl} mice, HF {n_mice_hf} mice)")

model = smf.mixedlm("body_weight ~ group * days_on_diet",
                    diet, groups=diet["mouse"])
result = model.fit(reml=True, method="lbfgs")

# Fixed-effects table
fx = pd.DataFrame({
    "term":     result.params.index,
    "estimate": result.params.values,
    "std_err":  result.bse.values,
    "z":        result.tvalues.values,
    "p_value":  result.pvalues.values,
    "ci_lo":    result.conf_int()[0].values,
    "ci_hi":    result.conf_int()[1].values,
})
fx.insert(0, "cable", CABLE)

# Random-effect variance + residual
re_var = float(result.cov_re.iloc[0, 0])
resid_var = float(result.scale)
fx_extra = pd.DataFrame({
    "cable": [CABLE, CABLE],
    "term":  ["random_intercept_var_(mouse)", "residual_var"],
    "estimate": [re_var, resid_var],
    "std_err":  [np.nan, np.nan],
    "z":        [np.nan, np.nan],
    "p_value":  [np.nan, np.nan],
    "ci_lo":    [np.nan, np.nan],
    "ci_hi":    [np.nan, np.nan],
})
fx_all = pd.concat([fx, fx_extra], ignore_index=True)

model_out = os.path.join(OUT_DIR, f"07_body_weight_model_{CABLE}.csv")
fx_all.to_csv(model_out, index=False)

print("\nFixed-effects table:")
print(fx.round(4).to_string(index=False))
print(f"\nSaved: {model_out}")
print("\nDone.")