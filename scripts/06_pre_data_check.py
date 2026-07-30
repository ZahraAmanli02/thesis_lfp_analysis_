# ============================================================
# 06_PRE_DATA_CHECK.PY
# Purpose:
# Pre-flight data check before running mixed-effects models in 06a.
# It loads 05a + 05b + 05c outputs, builds the master input table
# that 06a will use, and reports:
#   - group balance (HF vs CTRL)
#   - per-mouse recording counts (all recordings + diet-only)
#   - diet-phase distribution (baseline / diet / recovery)
#   - estrous-phase × group balance (full set + diet-only)
#   - body-weight distribution per group
#   - days_on_diet range per group
#   - NaN check on primary outcomes (log band powers + log ratios)
#   - skewness of log-transformed outcomes (model assumption check)
#
# Outputs:
#   - console summary printed to terminal
#   - 06_pre_data_check_<CABLE>.png  (6-panel visual overview)
#
# Why this exists:
# Mixed-effects models are sensitive to unbalanced design and to
# clusters (mice) with very few observations. This script flags
# such issues BEFORE 06a is run, so they can be discussed/justified
# in the thesis rather than discovered after model fitting.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/05b_band_ratios_<CABLE>/05b_band_ratios_<CABLE>.csv
#   outputs/05c_oscillation_episodes_<CABLE>/
#       05c_features_combined_with_episodes_<CABLE>.csv
#
# Output:
#   outputs/06_pre_data_check_<CABLE>/
#       06_pre_data_check_<CABLE>.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


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

OUT_DIR = os.path.join(OUTPUT_DIR, f"06_pre_data_check_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, f"06_pre_data_check_{CABLE}.png")


# Bands (must match 05a column scheme)
BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]

# Outcomes that 06a will use as primary
PRIMARY_LOG_POWERS = [f"log_{b}_abs" for b in BANDS]
PRIMARY_LOG_RATIOS = [
    "log_theta_delta", "log_low_gamma_theta", "log_high_gamma_theta",
    "log_fast_gamma_theta", "log_beta_theta",
]

# Colours
C_CTRL = "#4C72B0"
C_HF   = "#C44E52"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. LOAD + BUILD MASTER TABLE
# ------------------------------------------------------------
# 05a is the base (band powers + metadata). 05b log_* columns and
# 05c episode-feature columns are merged onto it via differential_file
# (1:1 because all three scripts already use the same clean set).
# log_<band>_abs columns are added here (statistical transform).
# ============================================================

if not os.path.exists(BAND_POWERS_PATH):
    raise FileNotFoundError(f"Missing 05a output:\n{BAND_POWERS_PATH}")
if not os.path.exists(BAND_RATIOS_PATH):
    raise FileNotFoundError(f"Missing 05b output:\n{BAND_RATIOS_PATH}")
if not os.path.exists(COMBINED_PATH):
    raise FileNotFoundError(f"Missing 05c output:\n{COMBINED_PATH}")

df_05a = pd.read_csv(BAND_POWERS_PATH)
df_05b = pd.read_csv(BAND_RATIOS_PATH)
df_05c = pd.read_csv(COMBINED_PATH)

# Add log band powers (statistical transform)
for band in BANDS:
    df_05a[f"log_{band}_abs"] = np.log10(df_05a[f"{band}_abs"])

# Merge log_* ratios from 05b
ratio_cols_to_merge = ["differential_file"] + PRIMARY_LOG_RATIOS
df = df_05a.merge(
    df_05b[ratio_cols_to_merge],
    on="differential_file", how="left", validate="one_to_one"
)

# Merge episode features from 05c
episode_cols = [c for c in df_05c.columns
                if any(c.startswith(f"{b}_") for b in
                       ["beta", "low_gamma", "high_gamma"])
                and any(c.endswith(suf) for suf in
                        ["_episode_rate", "_mean_duration_sec",
                         "_mean_amplitude", "_fraction_of_time"])]
df = df.merge(
    df_05c[["differential_file"] + episode_cols],
    on="differential_file", how="left", validate="one_to_one"
)

print(f"\nMaster table built: {df.shape[0]} recordings x {df.shape[1]} columns")

# Diet-phase only subset (what 06a will actually model)
diet_only = df[df["days_on_diet"].notna()].copy()


# ============================================================
# 3. CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("1. GROUP BALANCE")
print("=" * 60)
print(df.groupby("group").size().to_string())
print(f"Total: {len(df)} recordings, {df['mouse'].nunique()} mice")

print("\n" + "=" * 60)
print("2. RECORDINGS PER MOUSE (all 125)")
print("=" * 60)
mouse_counts = df.groupby(["group", "mouse"]).size().reset_index(name="n")
print(mouse_counts.to_string(index=False))

print("\n" + "=" * 60)
print("3. DIET PHASE BREAKDOWN")
print("=" * 60)
print(df.groupby(["group", "diet_phase"]).size().unstack(fill_value=0))

print("\n" + "=" * 60)
print("4. DIET-ONLY SUBSET (what 06a models will use)")
print("=" * 60)
print(f"Total: {len(diet_only)} recordings")
print(f"Mice with at least one diet recording: "
      f"{diet_only['mouse'].nunique()}")
print(f"\nPer mouse (diet recordings only):")
diet_mouse = diet_only.groupby(["group", "mouse"]).size().reset_index(name="n")
print(diet_mouse.to_string(index=False))
low_n = diet_mouse[diet_mouse["n"] < 3]
if len(low_n) > 0:
    print(f"\nWARN: {len(low_n)} mouse/mice with < 3 diet recordings — "
          f"random-intercept estimate will be weak for these.")

print("\n" + "=" * 60)
print("5. ESTROUS × GROUP (diet-only)")
print("=" * 60)
est_diet = (diet_only.groupby(["group", "estrous_phase"]).size()
            .unstack(fill_value=0))
print(est_diet)
low_cells = []
for grp in est_diet.index:
    for ph in est_diet.columns:
        if est_diet.loc[grp, ph] < 5:
            low_cells.append((grp, ph, est_diet.loc[grp, ph]))
if low_cells:
    print(f"\nWARN: estrous cells with n < 5:")
    for grp, ph, n in low_cells:
        print(f"  {grp} x {ph} = {n}  --> coefficient will be unstable")

print("\n" + "=" * 60)
print("6. BODY WEIGHT (per group)")
print("=" * 60)
print(df.groupby("group")["body_weight"]
      .describe()[["count", "mean", "std", "min", "max"]]
      .to_string())

print("\n" + "=" * 60)
print("7. days_on_diet RANGE (per group)")
print("=" * 60)
print(diet_only.groupby("group")["days_on_diet"]
      .describe()[["count", "mean", "min", "max"]]
      .to_string())

print("\n" + "=" * 60)
print("8. NaN CHECK ON PRIMARY OUTCOMES")
print("=" * 60)
all_primary = PRIMARY_LOG_POWERS + PRIMARY_LOG_RATIOS
any_nan = False
for col in all_primary:
    n_nan = df[col].isna().sum()
    flag = "  <-- WARN" if n_nan > 0 else ""
    print(f"  {col:30s}: {n_nan} NaN{flag}")
    if n_nan > 0:
        any_nan = True
if not any_nan:
    print("  --> all primary outcomes complete, no rows lost to NaN")

print("\n" + "=" * 60)
print("9. SKEWNESS AFTER LOG (model assumption)")
print("=" * 60)
print("  |skew| < 1   --> OK")
print("  |skew| < 2   --> acceptable")
print("  |skew| >= 2  --> consider revisiting transformation\n")
for col in all_primary:
    sk = stats.skew(df[col].dropna())
    flag = ""
    if abs(sk) >= 2:
        flag = "  <-- WARN"
    elif abs(sk) >= 1:
        flag = "  (acceptable)"
    print(f"  {col:30s}: skew = {sk:+.2f}{flag}")


# ============================================================
# 4. VISUAL OVERVIEW (6 panels)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# --- Panel A: recordings per mouse (all 125) ---
ax = axes[0, 0]
ctrl_m = mouse_counts[mouse_counts["group"] == "CTRL"]
hf_m   = mouse_counts[mouse_counts["group"] == "HF"]
x_c = np.arange(len(ctrl_m))
x_h = np.arange(len(ctrl_m), len(ctrl_m) + len(hf_m))
ax.bar(x_c, ctrl_m["n"], color=C_CTRL, alpha=0.85,
       label=f"CTRL ({len(ctrl_m)} mice)")
ax.bar(x_h, hf_m["n"], color=C_HF, alpha=0.85,
       label=f"HF ({len(hf_m)} mice)")
ax.set_xticks(np.concatenate([x_c, x_h]))
ax.set_xticklabels([f"M{m}" for m in pd.concat([ctrl_m["mouse"],
                                                hf_m["mouse"]])],
                   rotation=45, fontsize=8)
ax.set_ylabel("Recordings")
ax.set_title("A. Recordings per mouse (all clean)",
             fontweight="bold", loc="left")
ax.axhline(5, color="grey", ls="--", lw=0.6, alpha=0.5)
ax.legend(frameon=False, fontsize=9)

# --- Panel B: diet-only recordings per mouse ---
ax = axes[0, 1]
ctrl_d = diet_mouse[diet_mouse["group"] == "CTRL"]
hf_d   = diet_mouse[diet_mouse["group"] == "HF"]
x_c = np.arange(len(ctrl_d))
x_h = np.arange(len(ctrl_d), len(ctrl_d) + len(hf_d))
ax.bar(x_c, ctrl_d["n"], color=C_CTRL, alpha=0.85)
ax.bar(x_h, hf_d["n"], color=C_HF, alpha=0.85)
ax.set_xticks(np.concatenate([x_c, x_h]))
ax.set_xticklabels([f"M{m}" for m in pd.concat([ctrl_d["mouse"],
                                                hf_d["mouse"]])],
                   rotation=45, fontsize=8)
ax.set_ylabel("Diet recordings")
ax.set_title(f"B. Diet-only recordings (n={len(diet_only)})",
             fontweight="bold", loc="left")
ax.axhline(3, color="red", ls="--", lw=0.8, alpha=0.6)
ax.text(0.02, 0.95, "red = 3\n(below = weak random intercept)",
        transform=ax.transAxes, fontsize=8, va="top", color="red")

# --- Panel C: estrous × group (diet-only) ---
ax = axes[0, 2]
phases = ["A", "B", "C", "D"]
x = np.arange(len(phases))
w = 0.38
ctrl_vals = [est_diet.loc["CTRL", p] if p in est_diet.columns else 0
             for p in phases]
hf_vals   = [est_diet.loc["HF", p]   if p in est_diet.columns else 0
             for p in phases]
ax.bar(x - w/2, ctrl_vals, w, color=C_CTRL, alpha=0.85, label="CTRL")
ax.bar(x + w/2, hf_vals,   w, color=C_HF,   alpha=0.85, label="HF")
ax.set_xticks(x)
ax.set_xticklabels(phases)
ax.set_xlabel("Estrous phase")
ax.set_ylabel("Recordings")
ax.set_title("C. Estrous balance (diet-only)",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)
for i, (c, h) in enumerate(zip(ctrl_vals, hf_vals)):
    ax.text(i - w/2, c + 0.5, str(c), ha="center", fontsize=9)
    ax.text(i + w/2, h + 0.5, str(h), ha="center", fontsize=9,
            color="red" if h < 5 else "black",
            fontweight="bold" if h < 5 else "normal")

# --- Panel D: body weight distribution ---
ax = axes[1, 0]
for grp, col in [("CTRL", C_CTRL), ("HF", C_HF)]:
    mu = df[df["group"] == grp]["body_weight"].mean()
    ax.hist(df[df["group"] == grp]["body_weight"], bins=15,
            color=col, alpha=0.6, label=f"{grp} (mean={mu:.1f}g)")
ax.set_xlabel("Body weight (g)")
ax.set_ylabel("Recordings")
ax.set_title("D. Body weight (clear group separation)",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)

# --- Panel E: log_theta distribution ---
ax = axes[1, 1]
for grp, col in [("CTRL", C_CTRL), ("HF", C_HF)]:
    ax.hist(df[df["group"] == grp]["log_theta_abs"], bins=20,
            color=col, alpha=0.6, label=grp)
ax.set_xlabel("log10(theta absolute power)")
ax.set_ylabel("Recordings")
ax.set_title("E. log_theta distribution\n(normal-shaped = OK)",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)

# --- Panel F: theta vs days_on_diet (the modelled relationship) ---
ax = axes[1, 2]
for grp, col in [("CTRL", C_CTRL), ("HF", C_HF)]:
    sub = diet_only[diet_only["group"] == grp]
    ax.scatter(sub["days_on_diet"], sub["log_theta_abs"], color=col,
               alpha=0.5, s=30, label=grp, edgecolors="white",
               linewidths=0.5)
ax.set_xlabel("days_on_diet")
ax.set_ylabel("log10(theta abs)")
ax.set_title("F. theta vs time on diet (the model's question)",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)

fig.suptitle(f"Step 06 pre-flight data check ({CABLE}, "
             f"{len(df)} clean recordings)",
             fontsize=13, fontweight="bold", y=1.00)
fig.text(0.5, 0.005,
         f"Diet-phase subset (06a model input): {len(diet_only)} "
         f"recordings / {diet_only['mouse'].nunique()} mice "
         f"(CTRL {len(diet_only[diet_only['group']=='CTRL'])}/"
         f"{diet_only[diet_only['group']=='CTRL']['mouse'].nunique()}, "
         f"HF {len(diet_only[diet_only['group']=='HF'])}/"
         f"{diet_only[diet_only['group']=='HF']['mouse'].nunique()})",
         ha="center", fontsize=9, color="#444444", style="italic")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved overview figure:\n{OUT_PATH}")

print("\nPRE-FLIGHT CHECK finished successfully.")