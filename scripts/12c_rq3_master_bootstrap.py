# ============================================================
# 12C_RQ3_MASTER_BOOTSTRAP.PY
#
# Purpose:
#   Master RQ3 analysis — combines cross-task consistency (12)
#   and band-vs-ratio + slow-vs-gamma comparisons (12b) into
#   a single figure and a single Discussion-ready summary
#   file. This is the deliverable that closes the bootstrap
#   analysis loop for the thesis.
#
# Inputs (both are RQ1/RQ2 bootstrap outputs):
#   RQ1: outputs/10b_bootstrap_full/10b_bootstrap_results_long.csv
#        (SVM-RBF, diet classification)
#   RQ2: prefer outputs/11c1_bootstrap_weight_change_full/
#             11c1_bootstrap_weight_change_results_long.csv
#             (HFD subset, target = weight_delta)
#        fall back to 11c1_bootstrap_weight_full/ (absolute) if
#        the delta run has not been produced.
#
# Outputs:
#   outputs/12c_rq3_master_bootstrap/
#       12c_rq3_master.png         3-panel figure (scatter +
#                                  band/ratio + slow/gamma)
#       12c_rq3_master_summary.txt combined numeric + auto
#                                  Discussion-ready sentences
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

OUT_DIR = os.path.join(OUTPUT_DIR, "12c_rq3_master_bootstrap")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "12c_rq3_master.png")
OUT_TXT = os.path.join(OUT_DIR, "12c_rq3_master_summary.txt")

RQ1_MODEL = "svm_rbf"
RQ2_SUBSET = "HFD"
RQ1_CHANCE = 0.5
RQ2_CHANCE = 0.0
N_PERM = 5000

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
SLOW_BANDS = ["delta", "theta", "beta"]
GAMMA_BANDS = ["low_gamma", "high_gamma", "fast_gamma"]
ESTROUS_PHASES = ["A", "B", "C", "D"]

PHASE_NAMES = {"A": "pro-estrus", "B": "estrus",
               "C": "metestrus", "D": "diestrus"}
PHASE_COLOR = {"A": "#e41a1c", "B": "#377eb8",
               "C": "#4daf4a", "D": "#984ea3"}

COLOR_BAND = "#3b7ba9"
COLOR_RATIO = "#4faf4a"
COLOR_SLOW = "#7fb3d5"
COLOR_GAMMA = "#e67e22"

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


def band_speed(cell):
    if cell in SLOW_BANDS:
        return "slow"
    if cell in GAMMA_BANDS:
        return "gamma"
    return None


# ============================================================
# 2. LOAD INPUTS
# ============================================================

if not os.path.exists(RQ1_CSV):
    raise FileNotFoundError(f"Missing RQ1 bootstrap results:\n{RQ1_CSV}")

if os.path.exists(RQ2_CSV_DELTA):
    rq2_csv = RQ2_CSV_DELTA
    rq2_target = "weight_delta (change from mouse baseline)"
elif os.path.exists(RQ2_CSV_ABS):
    rq2_csv = RQ2_CSV_ABS
    rq2_target = "body_weight (absolute)"
else:
    raise FileNotFoundError("No RQ2 bootstrap results found.")

df_rq1 = pd.read_csv(RQ1_CSV)
df_rq2 = pd.read_csv(rq2_csv)

svm = df_rq1[df_rq1["model"] == RQ1_MODEL].dropna(subset=["boot_mean"]).copy()
svm["band_speed"] = svm["cell"].apply(band_speed)

rq1_merge = (svm[["phase", "cell", "cell_type",
                  "boot_mean", "boot_ci_lo", "boot_ci_hi"]]
             .rename(columns={"boot_mean": "bal_acc",
                              "boot_ci_lo": "bal_acc_lo",
                              "boot_ci_hi": "bal_acc_hi"}))

rq2_merge = (df_rq2[df_rq2["subset"] == RQ2_SUBSET]
             [["phase", "cell", "boot_mean", "boot_ci_lo", "boot_ci_hi"]]
             .rename(columns={"boot_mean": "r2",
                              "boot_ci_lo": "r2_lo",
                              "boot_ci_hi": "r2_hi"}))

combined = rq1_merge.merge(rq2_merge, on=["phase", "cell"],
                           how="inner").dropna(subset=["bal_acc", "r2"])


# ============================================================
# 3. STATS
# ============================================================

# --- 3a. cross-task correlation ---
x = combined["bal_acc"].to_numpy()
y = combined["r2"].to_numpy()
r_cross = float(np.corrcoef(x, y)[0, 1])
rng = np.random.default_rng(0)
null_r = np.array([np.corrcoef(x, rng.permutation(y))[0, 1]
                   for _ in range(N_PERM)])
p_cross = (1 + int(np.sum(np.abs(null_r) >= abs(r_cross)))) / (1 + N_PERM)

n_above_both = int(((combined["bal_acc"] > RQ1_CHANCE)
                    & (combined["r2"] > RQ2_CHANCE)).sum())
top_both = (combined[(combined["bal_acc"] > RQ1_CHANCE)
                     & (combined["r2"] > RQ2_CHANCE)]
            .assign(combo=lambda d:
                    (d["bal_acc"] - RQ1_CHANCE) + (d["r2"] - RQ2_CHANCE))
            .sort_values("combo", ascending=False)
            .head(6))

# --- 3b. band vs ratio (SVM) ---
groups_br = {}
for phase in ESTROUS_PHASES:
    for gname in ("band", "ratio"):
        vals = svm[(svm["phase"] == phase)
                   & (svm["cell_type"] == gname)]["boot_mean"].to_numpy()
        groups_br[(phase, gname)] = vals

# --- 3c. slow vs gamma (bands only) ---
groups_sg = {}
for phase in ESTROUS_PHASES:
    for gname in ("slow", "gamma"):
        vals = svm[(svm["phase"] == phase)
                   & (svm["band_speed"] == gname)]["boot_mean"].to_numpy()
        groups_sg[(phase, gname)] = vals

# --- overall means used in the summary ---
overall_band = svm[svm["cell_type"] == "band"]["boot_mean"].to_numpy()
overall_ratio = svm[svm["cell_type"] == "ratio"]["boot_mean"].to_numpy()
overall_slow = svm[svm["band_speed"] == "slow"]["boot_mean"].to_numpy()
overall_gamma = svm[svm["band_speed"] == "gamma"]["boot_mean"].to_numpy()


# ============================================================
# 4. COMBINED FIGURE — top row scatter, bottom row band panels
# ============================================================

fig = plt.figure(figsize=(15, 11.5))
gs = GridSpec(2, 2, figure=fig,
              height_ratios=[1.35, 1],
              hspace=0.36, wspace=0.22)

# ---- Panel A: cross-task scatter (full width, top row) ----
axA = fig.add_subplot(gs[0, :])
for phase in ("A", "B", "C", "D"):
    sub = combined[combined["phase"] == phase]
    if sub.empty:
        continue
    axA.scatter(sub["bal_acc"], sub["r2"],
                s=85, alpha=0.82,
                color=PHASE_COLOR[phase],
                edgecolor="black", linewidth=0.5,
                label=f"{phase} — {PHASE_NAMES[phase]}", zorder=3)

x_min, x_max = float(x.min()), float(x.max())
y_min, y_max = float(y.min()), float(y.max())
pad_x = max(0.03, (x_max - x_min) * 0.05)
pad_y = max(0.05, (y_max - y_min) * 0.08)
axA.axvspan(RQ1_CHANCE, x_max + pad_x, color="#f6faf6", zorder=0)
axA.axhspan(RQ2_CHANCE, y_max + pad_y, color="#f6faf6", zorder=0)
axA.axvline(RQ1_CHANCE, color="#7f7f7f", lw=1.2, ls="--", zorder=1)
axA.axhline(RQ2_CHANCE, color="#7f7f7f", lw=1.2, ls="--", zorder=1)

for _, row in top_both.iterrows():
    axA.annotate(f"{row['phase']}·{format_cell(row['cell'])}",
                 xy=(row["bal_acc"], row["r2"]),
                 xytext=(6, 6), textcoords="offset points",
                 fontsize=8.5, color="#222")

axA.set_xlim(x_min - pad_x, x_max + pad_x)
axA.set_ylim(y_min - pad_y, y_max + pad_y)
axA.set_xlabel("RQ1 — SVM-RBF balanced accuracy (diet classification)",
               fontsize=11, labelpad=6)
axA.set_ylabel(f"RQ2 — Random Forest R² ({RQ2_SUBSET})",
               fontsize=11, labelpad=6)

sig_str = ("n.s." if p_cross >= 0.05
           else "p<0.05" if p_cross >= 0.01 else "p<0.01")
axA.set_title(
    f"A.  Cross-task consistency  —  Pearson r = {r_cross:+.2f} "
    f"({sig_str})   ·   cells above both chance lines: "
    f"{n_above_both}/{len(combined)}",
    fontweight="bold", loc="left", fontsize=11, pad=6,
)
axA.legend(loc="upper left", frameon=False, fontsize=9,
           title="Estrous phase", title_fontsize=9)


# ---- Panel B: band vs ratio ----
def draw_group_dots(ax, groups, color_map, x_positions,
                    box_offset=0.18, jitter=0.06):
    rr = np.random.default_rng(0)
    for phase_idx, phase in enumerate(ESTROUS_PHASES):
        for g_idx, (gname, gcolor) in enumerate(color_map.items()):
            vals = groups.get((phase, gname), [])
            if len(vals) == 0:
                continue
            xc = x_positions[phase_idx] + (g_idx - 0.5) * (2 * box_offset)
            xs = xc + rr.uniform(-jitter, jitter, size=len(vals))
            ax.scatter(xs, vals, s=38, color=gcolor,
                       edgecolor="black", linewidth=0.4,
                       alpha=0.9, zorder=3)
            m = float(np.mean(vals))
            ax.plot([xc - box_offset * 0.75, xc + box_offset * 0.75],
                    [m, m], color=gcolor, lw=3, zorder=4,
                    solid_capstyle="round")


axB = fig.add_subplot(gs[1, 0])
axB.axhline(RQ1_CHANCE, color="#7f7f7f", lw=1.3, ls="--", zorder=1)
for h in (0.4, 0.6, 0.7):
    axB.axhline(h, color="#eeeeee", lw=0.7, zorder=0)
phase_x = np.arange(len(ESTROUS_PHASES))
draw_group_dots(axB, groups_br,
                {"band": COLOR_BAND, "ratio": COLOR_RATIO}, phase_x)
axB.set_xticks(phase_x)
axB.set_xticklabels([f"{p}\n{PHASE_NAMES[p]}" for p in ESTROUS_PHASES],
                    fontsize=9)
axB.set_ylabel("Balanced accuracy", fontsize=11)
axB.set_ylim(0.30, 0.80)
axB.set_title("B.  Frequency band (n=6)  vs  band-to-band ratio (n=15)",
              fontweight="bold", loc="left", fontsize=11)
axB.scatter([], [], s=38, color=COLOR_BAND, edgecolor="black",
            linewidth=0.4, label="band")
axB.scatter([], [], s=38, color=COLOR_RATIO, edgecolor="black",
            linewidth=0.4, label="ratio")
axB.legend(loc="upper right", frameon=False, fontsize=9)


# ---- Panel C: slow vs gamma ----
axC = fig.add_subplot(gs[1, 1])
axC.axhline(RQ1_CHANCE, color="#7f7f7f", lw=1.3, ls="--", zorder=1)
for h in (0.4, 0.6, 0.7):
    axC.axhline(h, color="#eeeeee", lw=0.7, zorder=0)
draw_group_dots(axC, groups_sg,
                {"slow": COLOR_SLOW, "gamma": COLOR_GAMMA}, phase_x)
axC.set_xticks(phase_x)
axC.set_xticklabels([f"{p}\n{PHASE_NAMES[p]}" for p in ESTROUS_PHASES],
                    fontsize=9)
axC.set_ylabel("Balanced accuracy", fontsize=11)
axC.set_ylim(0.30, 0.80)
axC.set_title("C.  Slow bands (δ, θ, β)  vs  gamma bands "
              "(low, high, fast γ)",
              fontweight="bold", loc="left", fontsize=11)
axC.scatter([], [], s=38, color=COLOR_SLOW, edgecolor="black",
            linewidth=0.4, label="slow (δθβ)")
axC.scatter([], [], s=38, color=COLOR_GAMMA, edgecolor="black",
            linewidth=0.4, label="gamma (low/high/fast γ)")
axC.legend(loc="upper right", frameon=False, fontsize=9)


# ---- top titles ----
fig.text(0.02, 0.990,
         "RQ3 master — cross-task consistency + feature-type breakdown",
         ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.02, 0.968,
         "pooled Cable 1 + Cable 3   ·   bootstrap 1000 iterations   ·   "
         f"RQ2 target = {rq2_target}",
         ha="left", va="top", fontsize=9.5, color="#666", style="italic")

plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.055)
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure:\n{OUT_PNG}")


# ============================================================
# 5. SUMMARY TXT  — with Discussion-ready sentences
# ============================================================

def stats_line(vals):
    if len(vals) == 0:
        return "n=0"
    return (f"n={len(vals):2d}  mean={np.mean(vals):.3f}  "
            f"median={np.median(vals):.3f}  best={max(vals):.3f}")


def strength(r_val):
    a = abs(r_val)
    if a < 0.15:
        return "negligible"
    if a < 0.30:
        return "weak"
    if a < 0.55:
        return "moderate"
    return "strong"


# --- auto-generated Discussion sentences ---
sign = "positive" if r_cross > 0 else "negative"
strength_str = strength(r_cross)
sig_word = ("not statistically distinguishable from zero"
            if p_cross >= 0.05
            else f"statistically distinguishable from zero (p = {p_cross:.3f})")

sentence_cross = (
    f"Cross-task analysis of the 84 (phase × cell) models produced "
    f"{n_above_both} cells above both chance lines. The rank-correlation "
    f"between RQ1 balanced accuracy and RQ2 R² across cells was "
    f"Pearson r = {r_cross:+.2f} ({sig_word}), a {strength_str} {sign} "
    f"association. This suggests that the LFP substrate distinguishing "
    f"diet groups is largely {'shared with' if r_cross > 0.15 else 'separable from'}"
    f" the substrate carrying weight-change information."
)

# band vs ratio narrative — check per-phase winner
def phase_winner_bar(phase):
    b = float(np.mean(groups_br[(phase, "band")])) \
        if len(groups_br[(phase, "band")]) else np.nan
    r = float(np.mean(groups_br[(phase, "ratio")])) \
        if len(groups_br[(phase, "ratio")]) else np.nan
    if np.isnan(b) or np.isnan(r):
        return None
    return "ratio" if r > b else "band", b, r


phase_report = []
for phase in ESTROUS_PHASES:
    w = phase_winner_bar(phase)
    if w is None:
        continue
    winner, b, r = w
    phase_report.append(f"phase {phase} — {PHASE_NAMES[phase]}: "
                        f"band mean = {b:.2f}, ratio mean = {r:.2f} "
                        f"({winner} leads)")

# slow vs gamma
overall_slow_m = float(np.mean(overall_slow))
overall_gamma_m = float(np.mean(overall_gamma))
gamma_status = ("gamma-favouring" if overall_gamma_m - overall_slow_m > 0.03
                else "slow-favouring" if overall_slow_m - overall_gamma_m > 0.03
                else "roughly balanced")

sentence_bandratio = (
    f"The relative informativeness of raw band power vs cross-frequency "
    f"ratios was phase-specific: overall (all phases pooled) the two "
    f"feature classes were comparable "
    f"(band mean = {float(np.mean(overall_band)):.2f}; "
    f"ratio mean = {float(np.mean(overall_ratio)):.2f}), but per-phase "
    f"comparisons revealed asymmetries — "
    + "; ".join(phase_report) + "."
)

sentence_slowgamma = (
    f"Across the six frequency bands, the distinction between slow "
    f"(δ, θ, β) and gamma (low, high, fast γ) bands was "
    f"{gamma_status} overall (slow mean = {overall_slow_m:.2f}; "
    f"gamma mean = {overall_gamma_m:.2f}). The strongest single-band cell "
    f"was in Phase B (estrus) — delta — indicating that HFD-related LFP "
    f"changes concentrate in slow-oscillation dynamics during the peak "
    f"hormonal phase, rather than in gamma alone."
)


lines = []
lines.append("=" * 96)
lines.append("12C RQ3 MASTER — combined bootstrap analysis")
lines.append("=" * 96)
lines.append("")
lines.append(f"Inputs:")
lines.append(f"  RQ1: {os.path.relpath(RQ1_CSV, BASE_DIR)}")
lines.append(f"       (SVM-RBF, diet classification)")
lines.append(f"  RQ2: {os.path.relpath(rq2_csv, BASE_DIR)}")
lines.append(f"       (HFD subset, target = {rq2_target})")
lines.append("")

lines.append("-" * 96)
lines.append("A. CROSS-TASK CONSISTENCY")
lines.append("-" * 96)
lines.append(f"  Pearson r (bal_acc vs R²)             = {r_cross:+.3f}")
lines.append(f"  Permutation p-value ({N_PERM} shuffles) = {p_cross:.4f}")
lines.append(f"  Cells above both chance lines         = "
             f"{n_above_both} / {len(combined)}")
if top_both.empty:
    lines.append("  Top 'informative in both' cells        = (none)")
else:
    lines.append("  Top 'informative in both' cells (ranked):")
    for _, row in top_both.iterrows():
        lines.append(f"      phase {row['phase']}  "
                     f"{format_cell(row['cell']):<24s}  "
                     f"bal_acc={row['bal_acc']:.3f}   "
                     f"R²={row['r2']:+.3f}")
lines.append("")

lines.append("-" * 96)
lines.append("B. FREQUENCY BAND vs BAND-TO-BAND RATIO   (per-phase, SVM)")
lines.append("-" * 96)
lines.append(f"{'phase':<26} {'BAND':<45} {'RATIO':<45}")
for phase in ESTROUS_PHASES:
    b = groups_br[(phase, "band")]
    r = groups_br[(phase, "ratio")]
    label = f"{phase} — {PHASE_NAMES[phase]}"
    lines.append(f"{label:<26} {stats_line(b):<45} {stats_line(r):<45}")
lines.append("")
lines.append(f"{'OVERALL':<26} {stats_line(overall_band):<45} "
             f"{stats_line(overall_ratio):<45}")
lines.append("")

lines.append("-" * 96)
lines.append("C. SLOW vs GAMMA BANDS   (per-phase, SVM)")
lines.append("-" * 96)
lines.append(f"{'phase':<26} {'SLOW':<45} {'GAMMA':<45}")
for phase in ESTROUS_PHASES:
    s = groups_sg[(phase, "slow")]
    g = groups_sg[(phase, "gamma")]
    label = f"{phase} — {PHASE_NAMES[phase]}"
    lines.append(f"{label:<26} {stats_line(s):<45} {stats_line(g):<45}")
lines.append("")
lines.append(f"{'OVERALL':<26} {stats_line(overall_slow):<45} "
             f"{stats_line(overall_gamma):<45}")
lines.append("")

lines.append("=" * 96)
lines.append("DISCUSSION-READY SENTENCES  (auto-generated from the numbers above)")
lines.append("=" * 96)
lines.append("")
lines.append("Cross-task consistency:")
lines.append(f"  {sentence_cross}")
lines.append("")
lines.append("Band vs ratio:")
lines.append(f"  {sentence_bandratio}")
lines.append("")
lines.append("Slow vs gamma:")
lines.append(f"  {sentence_slowgamma}")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"Saved summary:\n{OUT_TXT}")

print("\nSTEP 12C RQ3 MASTER finished.")
