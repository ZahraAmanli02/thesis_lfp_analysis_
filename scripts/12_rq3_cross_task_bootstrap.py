# ============================================================
# 12_RQ3_CROSS_TASK_BOOTSTRAP.PY
#
# Purpose:
#   RQ3 — cross-task consistency between diet classification
#   (RQ1) and body-weight-change regression (RQ2).
#
#   The RQ3 question: do the SAME (phase × cell) combinations
#   that classify diet also predict weight change? A shared
#   substrate would show up as an upper-right cluster (or a
#   positive correlation) in the scatter of RQ1 vs RQ2 means.
#
# Inputs:
#   RQ1: outputs/10b_bootstrap_full/10b_bootstrap_results_long.csv
#        (uses SVM-RBF, the primary classifier)
#   RQ2: prefer outputs/11c1_bootstrap_weight_change_full/
#        11c1_bootstrap_weight_change_results_long.csv (delta target)
#        fall back to 11c1_bootstrap_weight_full/ (absolute target)
#        if the delta run has not been produced yet
#        (uses HFD subset — where the weight-change signal lives)
#
# Outputs:
#   outputs/12_rq3_cross_task_bootstrap/
#       12_cross_task_scatter.png
#       12_cross_task_summary.txt
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RQ1_CSV = os.path.join(
    OUTPUT_DIR, "10b_bootstrap_full", "10b_bootstrap_results_long.csv"
)
RQ2_CSV_DELTA = os.path.join(
    OUTPUT_DIR, "11c1_bootstrap_weight_change_full",
    "11c1_bootstrap_weight_change_results_long.csv"
)
RQ2_CSV_ABS = os.path.join(
    OUTPUT_DIR, "11c1_bootstrap_weight_full",
    "11c1_bootstrap_weight_results_long.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, "12_rq3_cross_task_bootstrap")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "12_cross_task_scatter.png")
OUT_TXT = os.path.join(OUT_DIR, "12_cross_task_summary.txt")

RQ1_MODEL = "svm_rbf"
RQ2_SUBSET = "HFD"
RQ1_CHANCE = 0.5
RQ2_CHANCE = 0.0

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]

PHASE_NAMES = {"A": "pro-estrus", "B": "estrus",
               "C": "metestrus", "D": "diestrus"}
PHASE_COLOR = {"A": "#e41a1c", "B": "#377eb8",
               "C": "#4daf4a", "D": "#984ea3"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def format_cell(cell):
    if cell in BANDS:
        return cell
    for b in sorted(BANDS, key=len, reverse=True):
        if cell.startswith(b + "_"):
            rest = cell[len(b) + 1:]
            if rest in BANDS:
                return f"{b} / {rest}"
    return cell


# ============================================================
# 2. LOAD RQ1 AND RQ2 RESULTS
# ============================================================

if not os.path.exists(RQ1_CSV):
    raise FileNotFoundError(f"Missing RQ1 bootstrap results:\n{RQ1_CSV}")

if os.path.exists(RQ2_CSV_DELTA):
    rq2_csv = RQ2_CSV_DELTA
    rq2_target_label = "weight_delta (change from mouse baseline)"
elif os.path.exists(RQ2_CSV_ABS):
    rq2_csv = RQ2_CSV_ABS
    rq2_target_label = "body_weight (absolute)"
    print("Note: delta results not found, falling back to absolute weight.")
else:
    raise FileNotFoundError(
        "No RQ2 bootstrap results found. Expected one of:\n"
        f"  {RQ2_CSV_DELTA}\n"
        f"  {RQ2_CSV_ABS}"
    )

df_rq1 = pd.read_csv(RQ1_CSV)
df_rq2 = pd.read_csv(rq2_csv)

rq1 = (df_rq1[df_rq1["model"] == RQ1_MODEL]
       [["phase", "cell", "cell_type",
         "boot_mean", "boot_ci_lo", "boot_ci_hi"]]
       .rename(columns={"boot_mean": "bal_acc",
                        "boot_ci_lo": "bal_acc_lo",
                        "boot_ci_hi": "bal_acc_hi"}))

rq2 = (df_rq2[df_rq2["subset"] == RQ2_SUBSET]
       [["phase", "cell",
         "boot_mean", "boot_ci_lo", "boot_ci_hi"]]
       .rename(columns={"boot_mean": "r2",
                        "boot_ci_lo": "r2_lo",
                        "boot_ci_hi": "r2_hi"}))

combined = rq1.merge(rq2, on=["phase", "cell"], how="inner").dropna(
    subset=["bal_acc", "r2"]
)

print(f"RQ1 rows loaded: {len(rq1)}")
print(f"RQ2 rows loaded: {len(rq2)}")
print(f"Merged (phase × cell) pairs with both metrics: {len(combined)}")


# ============================================================
# 3. STATS — Pearson correlation between the two metrics
# ============================================================

if len(combined) < 3:
    raise RuntimeError("Not enough merged rows to compute a correlation.")

x = combined["bal_acc"].to_numpy()
y = combined["r2"].to_numpy()
r = float(np.corrcoef(x, y)[0, 1])

# a p-value from a simple permutation test on r
rng = np.random.default_rng(0)
N_PERM = 5000
null_r = np.empty(N_PERM)
for i in range(N_PERM):
    null_r[i] = np.corrcoef(x, rng.permutation(y))[0, 1]
p_val = (1 + int(np.sum(np.abs(null_r) >= abs(r)))) / (1 + N_PERM)

# "informative in both" quadrant: bal_acc > 0.5 AND r2 > 0
both = combined[(combined["bal_acc"] > RQ1_CHANCE)
                & (combined["r2"] > RQ2_CHANCE)]

# cells whose 95% CI for both metrics clears their chance line
both_ci = combined[(combined["bal_acc_lo"] > RQ1_CHANCE)
                   & (combined["r2_lo"] > RQ2_CHANCE)]


# ============================================================
# 4. SCATTER PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(11, 8))

for phase in ("A", "B", "C", "D"):
    sub = combined[combined["phase"] == phase]
    if sub.empty:
        continue
    ax.scatter(sub["bal_acc"], sub["r2"],
               s=90, alpha=0.82,
               color=PHASE_COLOR[phase],
               edgecolor="black", linewidth=0.5,
               label=f"{phase} — {PHASE_NAMES[phase]}", zorder=3)

# chance reference lines
ax.axvline(RQ1_CHANCE, color="#7f7f7f", lw=1.2, ls="--", zorder=1)
ax.axhline(RQ2_CHANCE, color="#7f7f7f", lw=1.2, ls="--", zorder=1)

# shade the "informative in both" quadrant subtly
x_max, y_max = float(x.max()), float(y.max())
x_min, y_min = float(x.min()), float(y.min())
pad_x = max(0.03, (x_max - x_min) * 0.05)
pad_y = max(0.05, (y_max - y_min) * 0.08)
ax.axvspan(RQ1_CHANCE, x_max + pad_x, ymin=0, ymax=1,
           color="#f6faf6", zorder=0)
ax.axhspan(RQ2_CHANCE, y_max + pad_y, xmin=0, xmax=1,
           color="#f6faf6", zorder=0)

# annotate top cells in the "informative in both" quadrant
top_both = both.assign(
    combo=lambda d: (d["bal_acc"] - RQ1_CHANCE) + (d["r2"] - RQ2_CHANCE)
).sort_values("combo", ascending=False).head(6)
for _, row in top_both.iterrows():
    ax.annotate(
        f"{row['phase']}·{format_cell(row['cell'])}",
        xy=(row["bal_acc"], row["r2"]),
        xytext=(6, 6), textcoords="offset points",
        fontsize=8.5, color="#222",
        arrowprops=None,
    )

# axis labels + title
ax.set_xlim(x_min - pad_x, x_max + pad_x)
ax.set_ylim(y_min - pad_y, y_max + pad_y)
ax.set_xlabel("RQ1 — SVM-RBF balanced accuracy (diet classification)",
              fontsize=11, labelpad=8)
ax.set_ylabel(f"RQ2 — Random Forest R² ({RQ2_SUBSET}, "
              f"{rq2_target_label.split(' (')[0]})",
              fontsize=11, labelpad=8)

fig.text(0.02, 0.97,
         "RQ3 — cross-task consistency",
         ha="left", va="top", fontsize=15, fontweight="bold")
sig_str = "n.s." if p_val >= 0.05 else "p<0.05" if p_val >= 0.01 else "p<0.01"
fig.text(0.02, 0.935,
         f"pooled Cable 1 + Cable 3   ·   Pearson r = {r:+.2f}   "
         f"({sig_str}, permutation)   ·   "
         f"cells above both chance lines: {len(both)}/{len(combined)}",
         ha="left", va="top", fontsize=10, color="#666", style="italic")

ax.legend(loc="upper left", frameon=False, fontsize=9,
          title="Estrous phase", title_fontsize=9)

plt.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.09)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure:\n{OUT_PNG}")


# ============================================================
# 5. INTERPRETATION SUMMARY (TXT)
# ============================================================

def interpret_correlation(r_val, p_val):
    strength = ("negligible" if abs(r_val) < 0.15
                else "weak" if abs(r_val) < 0.30
                else "moderate" if abs(r_val) < 0.55
                else "strong")
    sig = ("not statistically distinguishable from zero"
           if p_val >= 0.05
           else f"statistically distinguishable from zero (p = {p_val:.3f})")
    sign = "positive" if r_val > 0 else "negative"
    if abs(r_val) < 0.15:
        return f"correlation is {strength} ({sig})"
    return f"{strength} {sign} correlation ({sig})"


lines = []
lines.append("=" * 88)
lines.append("12 RQ3 — CROSS-TASK CONSISTENCY  (bootstrap-based)")
lines.append("=" * 88)
lines.append("")
lines.append("Question:")
lines.append("  Do the same (phase × cell) combinations that classify diet")
lines.append("  also predict weight change? If yes -> shared LFP substrate.")
lines.append("")
lines.append("Inputs:")
lines.append(f"  RQ1: {os.path.relpath(RQ1_CSV, BASE_DIR)}")
lines.append(f"       (SVM-RBF, primary classifier)")
lines.append(f"  RQ2: {os.path.relpath(rq2_csv, BASE_DIR)}")
lines.append(f"       (HFD subset, target = {rq2_target_label})")
lines.append("")
lines.append(f"Cells with both metrics available: {len(combined)}")
lines.append("")

lines.append("-" * 88)
lines.append("HEADLINE STATISTIC")
lines.append("-" * 88)
lines.append(f"  Pearson r (bal_acc vs R²) = {r:+.3f}")
lines.append(f"  Permutation p-value      = {p_val:.4f}  ({N_PERM} shuffles)")
lines.append(f"  Interpretation           = {interpret_correlation(r, p_val)}")
lines.append("")
lines.append(f"  Cells above both chance lines (bal_acc > 0.5 AND R² > 0):")
lines.append(f"    {len(both)} / {len(combined)}")
lines.append(f"  Cells whose 95% CI clears both chance lines (stricter):")
lines.append(f"    {len(both_ci)} / {len(combined)}")
lines.append("")

lines.append("-" * 88)
lines.append("TOP CELLS INFORMATIVE FOR BOTH TASKS  (ranked by combined lift)")
lines.append("-" * 88)
if top_both.empty:
    lines.append("  (none; no cells sit above both chance lines)")
else:
    lines.append(f"  {'phase':<6} {'cell':<24} {'bal_acc':>9} {'R²':>8}")
    for _, row in top_both.iterrows():
        lines.append(f"  {row['phase']:<6} "
                     f"{format_cell(row['cell']):<24} "
                     f"{row['bal_acc']:>9.3f} "
                     f"{row['r2']:>+8.3f}")
lines.append("")

lines.append("-" * 88)
lines.append("HOW TO READ THIS")
lines.append("-" * 88)
lines.append("  * Upper-right quadrant = cells informative in BOTH tasks.")
lines.append("    A cluster there means diet and weight are picked up by the")
lines.append("    same LFP signature — a shared physiological substrate.")
lines.append("  * A high positive Pearson r extends the same idea: the ranking")
lines.append("    of cells by diet-classification tracks the ranking by")
lines.append("    weight-change regression.")
lines.append("  * Wide scatter / r ≈ 0 = the two tasks pick up different LFP")
lines.append("    features. This is itself informative: it means diet and")
lines.append("    weight change are reflected in DIFFERENT oscillation regimes.")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved summary:\n{OUT_TXT}")

print("\nSTEP 12 RQ3 CROSS-TASK finished.")
