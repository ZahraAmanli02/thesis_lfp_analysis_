# ============================================================
# 14_RQ3_FEATURE_RELEVANCE.PY
#
# Deep-dive for RQ3 — "which frequency bands and estrous phases
# carry the diet signal?"
#
# RQ3 is different from RQ1 / RQ2: it does not have a single
# per-model metric. Its evaluation is about PATTERNS across all
# 84 (phase x cell) models. This script computes and visualises:
#
#   1. Ranking: top-20 informative cells across both tasks.
#   2. Phase informativeness — which estrous phase is most
#      informative for diet? for weight?
#   3. Band vs ratio comparison — does the signal live in raw
#      band power or in cross-frequency ratios?
#   4. Cross-task consistency — do cells that predict diet also
#      predict weight?  (This is a genuine RQ3 test.)
#   5. Gamma-vs-slow band breakdown — is the signal in gamma
#      (as the LH literature suggests), or elsewhere?
#   6. Auto-generated interpretation.
#
# Inputs:
#   outputs/10b_classify_group_<CABLE>/10b_results_long_<CABLE>.csv
#   outputs/10c_regress_weight_<CABLE>/10c_results_long_<CABLE>.csv
#
# Outputs:
#   outputs/14_rq3_feature_relevance_<CABLE>/
#     14_ranking_top_cells_<CABLE>.png       top 20 cells, both tasks
#     14_phase_informativeness_<CABLE>.png   per-phase summary
#     14_band_vs_ratio_<CABLE>.png           band vs ratio comparison
#     14_cross_task_consistency_<CABLE>.png  RQ1 x RQ2 scatter per cell
#     14_gamma_vs_slow_<CABLE>.png           gamma vs delta/theta breakdown
#     14_INTERPRETATION_<CABLE>.txt          plain-language summary
#
# Usage:
#   Set CABLE = "Cable1" or "Cable3", run twice.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"
ALPHA = 0.05

BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma", "fast_gamma"]
RATIOS = [
    "theta_delta", "beta_delta", "low_gamma_delta",
    "high_gamma_delta", "fast_gamma_delta",
    "beta_theta", "low_gamma_theta", "high_gamma_theta", "fast_gamma_theta",
    "low_gamma_beta", "high_gamma_beta", "fast_gamma_beta",
    "high_gamma_low_gamma", "fast_gamma_low_gamma", "fast_gamma_high_gamma",
]
CELLS = BANDS + RATIOS
ESTROUS_PHASES = ["A", "B", "C", "D"]
SLOW_BANDS = ["delta", "theta", "beta"]
GAMMA_BANDS = ["low_gamma", "high_gamma", "fast_gamma"]

COLOR_BAND = "#2C6E9B"
COLOR_RATIO = "#4C8C4A"
COLOR_SLOW = "#7FB3D5"
COLOR_GAMMA = "#E67E22"
COLOR_DIET = "#1F4E79"
COLOR_WEIGHT = "#8E44AD"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CSV_10B = os.path.join(
    OUTPUT_DIR, f"10b_classify_group_{CABLE}", f"10b_results_long_{CABLE}.csv"
)
CSV_10C = os.path.join(
    OUTPUT_DIR, f"10c_regress_weight_{CABLE}", f"10c_results_long_{CABLE}.csv"
)
OUT_DIR = os.path.join(OUTPUT_DIR, f"14_rq3_feature_relevance_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "figure.facecolor": "white",
})


# ============================================================
# 2. LOAD RESULTS
# ============================================================

print(f"\n{'=' * 70}")
print(f"14 RQ3 FEATURE RELEVANCE — {CABLE}")
print("=" * 70)

if not os.path.exists(CSV_10B):
    raise FileNotFoundError(f"Missing 10b results:\n{CSV_10B}")
if not os.path.exists(CSV_10C):
    raise FileNotFoundError(f"Missing 10c results:\n{CSV_10C}")

df_b = pd.read_csv(CSV_10B)
df_c = pd.read_csv(CSV_10C)


def get_metric(df, model, metric):
    sub = df[(df["model"] == model) & (df["metric"] == metric)]
    return sub[["phase", "cell", "value"]].dropna().rename(columns={"value": metric})


diet = (get_metric(df_b, "svm_rbf", "balanced_accuracy")
        .merge(get_metric(df_b, "svm_rbf", "perm_p_value_balanced_accuracy"),
               on=["phase", "cell"], how="left")
        .rename(columns={"balanced_accuracy": "bal_acc",
                         "perm_p_value_balanced_accuracy": "p_diet"}))

weight = (get_metric(df_c, "random_forest", "r2")
          .merge(get_metric(df_c, "random_forest", "perm_p_value_r2"),
                 on=["phase", "cell"], how="left")
          .rename(columns={"r2": "r2",
                           "perm_p_value_r2": "p_weight"}))

combined = diet.merge(weight, on=["phase", "cell"], how="outer")
combined["cell_type"] = combined["cell"].apply(
    lambda c: "band" if c in BANDS else ("ratio" if c in RATIOS else "?")
)

n_total = len(combined)
n_sig_diet = int((combined["p_diet"] < ALPHA).sum())
n_sig_weight = int((combined["p_weight"] < ALPHA).sum())
print(f"Loaded {n_total} (phase x cell) pairs.")
print(f"Diet-significant  : {n_sig_diet} / {n_total}")
print(f"Weight-significant: {n_sig_weight} / {n_total}")


# ============================================================
# 3. FIGURE 1 — TOP-20 RANKING (both tasks side-by-side)
# ============================================================

top_diet = combined.dropna(subset=["bal_acc"]).sort_values(
    "bal_acc", ascending=False).head(20)
top_weight = combined.dropna(subset=["r2"]).sort_values(
    "r2", ascending=False).head(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- diet ---
ax = axes[0]
ys = np.arange(len(top_diet))
colors_diet = [COLOR_BAND if r["cell_type"] == "band" else COLOR_RATIO
               for _, r in top_diet.iterrows()]
ax.barh(ys, top_diet["bal_acc"], color=colors_diet, edgecolor="black")
ax.axvline(0.5, color="grey", linestyle="--", lw=1.5, label="chance = 0.5")
labels = [f"phase {r['phase']} | {r['cell']}" for _, r in top_diet.iterrows()]
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
# Numeric value next to every bar + red star if significant
for y, (_, r) in zip(ys, top_diet.iterrows()):
    if pd.isna(r["bal_acc"]):
        continue
    star = " ★" if (not pd.isna(r["p_diet"]) and r["p_diet"] < ALPHA) else ""
    ax.text(r["bal_acc"] + 0.01, y, f"{r['bal_acc']:.2f}{star}",
            color="#C0392B" if star else "black",
            fontsize=9, va="center", ha="left", fontweight="bold")
ax.set_xlim(0.3, 1.0)
ax.set_xlabel("Balanced accuracy", fontsize=12, fontweight="bold")
ax.set_title("A. Top-20 cells for DIET classification",
             fontsize=13, fontweight="bold", loc="left")
ax.legend(fontsize=10, loc="lower right"); ax.grid(axis="x", alpha=0.3)

# --- weight ---
ax = axes[1]
ys = np.arange(len(top_weight))
colors_wt = [COLOR_BAND if r["cell_type"] == "band" else COLOR_RATIO
             for _, r in top_weight.iterrows()]
ax.barh(ys, top_weight["r2"], color=colors_wt, edgecolor="black")
ax.axvline(0.0, color="grey", linestyle="--", lw=1.5, label="chance R² = 0")
labels = [f"phase {r['phase']} | {r['cell']}" for _, r in top_weight.iterrows()]
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
# Numeric value next to every bar (so tiny / near-zero bars are still labelled)
for y, (_, r) in zip(ys, top_weight.iterrows()):
    if pd.isna(r["r2"]):
        continue
    txt_x = r["r2"] + (0.02 if r["r2"] >= 0 else -0.02)
    ha = "left" if r["r2"] >= 0 else "right"
    star = " ★" if (not pd.isna(r["p_weight"]) and r["p_weight"] < ALPHA) else ""
    ax.text(txt_x, y, f"{r['r2']:.2f}{star}",
            color="#C0392B" if star else "black",
            fontsize=9, va="center", ha=ha, fontweight="bold")
ax.set_xlabel("R²", fontsize=12, fontweight="bold")
ax.set_title("B. Top-20 cells for WEIGHT regression",
             fontsize=13, fontweight="bold", loc="left")
ax.legend(fontsize=10, loc="lower right"); ax.grid(axis="x", alpha=0.3)

# shared legend for band vs ratio
handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLOR_BAND, label="Frequency band"),
    plt.Rectangle((0, 0), 1, 1, color=COLOR_RATIO, label="Band-to-band ratio"),
    plt.Line2D([], [], marker="*", linestyle="None", color="#C0392B",
               markersize=10, label="★ = permutation p < 0.05"),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"RQ3 — Ranking of the most informative cells   |   {CABLE}",
             fontsize=15, fontweight="bold", y=1.00)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, f"14_ranking_top_cells_{CABLE}.png")
plt.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  saved: {out1}")


# ============================================================
# 4. FIGURE 2 — PHASE INFORMATIVENESS
# ============================================================

phase_summary = []
for phase in ESTROUS_PHASES:
    sub_d = combined[combined["phase"] == phase]
    sub_w = combined[combined["phase"] == phase]
    phase_summary.append({
        "phase": phase,
        "diet_n_sig":     int((sub_d["p_diet"] < ALPHA).sum()),
        "diet_mean":      float(sub_d["bal_acc"].mean()),
        "diet_max":       float(sub_d["bal_acc"].max()),
        "weight_n_sig":   int((sub_w["p_weight"] < ALPHA).sum()),
        "weight_mean":    float(sub_w["r2"].mean()),
        "weight_max":     float(sub_w["r2"].max()),
    })
ph_df = pd.DataFrame(phase_summary)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# panel A: significant-cell count
ax = axes[0]
xs = np.arange(len(ESTROUS_PHASES))
w = 0.38
ax.bar(xs - w/2, ph_df["diet_n_sig"], w, color=COLOR_DIET,
       edgecolor="black", label="Diet")
ax.bar(xs + w/2, ph_df["weight_n_sig"], w, color=COLOR_WEIGHT,
       edgecolor="black", label="Weight")
ax.set_xticks(xs); ax.set_xticklabels(ESTROUS_PHASES, fontweight="bold")
ax.set_xlabel("Estrous phase", fontsize=11, fontweight="bold")
ax.set_ylabel("Number of significant cells (p < 0.05)",
              fontsize=11, fontweight="bold")
ax.set_title("A. Significant cells per phase",
             fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)

# panel B: mean metric per phase
# Bars are drawn as (value - chance) around a baseline at chance, so
# a phase performing BELOW chance shows as a red downward bar —
# clearly visible instead of invisible.
ax = axes[1]
chance_diet = 0.5
diet_delta = ph_df["diet_mean"] - chance_diet
diet_colors = [COLOR_DIET if d >= 0 else "#C0392B" for d in diet_delta]
ax.bar(xs, diet_delta, color=diet_colors, edgecolor="black",
       bottom=chance_diet)
ax.axhline(chance_diet, color="grey", linestyle="--", lw=1.5, label="chance 0.5")
for x, val in zip(xs, ph_df["diet_mean"]):
    if pd.isna(val):
        ax.text(x, chance_diet, "NaN", ha="center", va="center",
                fontsize=10, color="#888", style="italic")
    else:
        ax.text(x, val + 0.005 if val >= chance_diet else val - 0.008,
                f"{val:.2f}", ha="center",
                va="bottom" if val >= chance_diet else "top",
                fontsize=10, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(ESTROUS_PHASES, fontweight="bold")
ax.set_xlabel("Estrous phase", fontsize=11, fontweight="bold")
ax.set_ylabel("Mean balanced accuracy", fontsize=11, fontweight="bold")
ax.set_title("B. Mean diet-accuracy per phase\n(Red = below chance)",
             fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)

# panel C: mean R2 per phase (same trick — red for below chance)
ax = axes[2]
r2_colors = [COLOR_WEIGHT if r >= 0 else "#C0392B" for r in ph_df["weight_mean"]]
ax.bar(xs, ph_df["weight_mean"], color=r2_colors, edgecolor="black")
ax.axhline(0.0, color="grey", linestyle="--", lw=1.5, label="chance R²=0")
for x, val in zip(xs, ph_df["weight_mean"]):
    if pd.isna(val):
        ax.text(x, 0, "NaN", ha="center", va="center",
                fontsize=10, color="#888", style="italic")
    else:
        ax.text(x, val + 0.02 if val >= 0 else val - 0.02,
                f"{val:.2f}", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=10, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(ESTROUS_PHASES, fontweight="bold")
ax.set_xlabel("Estrous phase", fontsize=11, fontweight="bold")
ax.set_ylabel("Mean R²", fontsize=11, fontweight="bold")
ax.set_title("C. Mean weight-R² per phase\n(Red = below chance)",
             fontsize=12, fontweight="bold", loc="left")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)

fig.suptitle(f"RQ3 — Which estrous phase is most informative?   |   {CABLE}",
             fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, f"14_phase_informativeness_{CABLE}.png")
plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  saved: {out2}")


# ============================================================
# 5. FIGURE 3 — BAND vs RATIO
# ============================================================

bvr = {"band": {}, "ratio": {}}
for typ in ["band", "ratio"]:
    sub = combined[combined["cell_type"] == typ]
    bvr[typ] = {
        "diet_n_sig":   int((sub["p_diet"] < ALPHA).sum()),
        "diet_n":       len(sub),
        "diet_mean":    float(sub["bal_acc"].mean()),
        "diet_max":     float(sub["bal_acc"].max()),
        "weight_n_sig": int((sub["p_weight"] < ALPHA).sum()),
        "weight_n":     len(sub),
        "weight_mean":  float(sub["r2"].mean()),
        "weight_max":   float(sub["r2"].max()),
    }

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
xs = [0, 1]

# panel A: diet
ax = axes[0]
means = [bvr["band"]["diet_mean"], bvr["ratio"]["diet_mean"]]
maxes = [bvr["band"]["diet_max"],  bvr["ratio"]["diet_max"]]
ax.bar([x - 0.2 for x in xs], means, 0.38,
       color=[COLOR_BAND, COLOR_RATIO], edgecolor="black", label="mean")
ax.bar([x + 0.2 for x in xs], maxes, 0.38,
       color=[COLOR_BAND, COLOR_RATIO], edgecolor="black", hatch="//",
       label="max")
ax.axhline(0.5, color="grey", linestyle="--", lw=1.5, label="chance")
ax.set_xticks(xs); ax.set_xticklabels(["Bands (n=24)", "Ratios (n=60)"],
                                      fontsize=11, fontweight="bold")
ax.set_ylabel("Balanced accuracy", fontsize=11, fontweight="bold")
ax.set_title(
    f"A. Diet: bands {bvr['band']['diet_n_sig']}/24 sig, "
    f"ratios {bvr['ratio']['diet_n_sig']}/60 sig",
    fontsize=11, fontweight="bold", loc="left"
)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

# panel B: weight
ax = axes[1]
means = [bvr["band"]["weight_mean"], bvr["ratio"]["weight_mean"]]
maxes = [bvr["band"]["weight_max"],  bvr["ratio"]["weight_max"]]
ax.bar([x - 0.2 for x in xs], means, 0.38,
       color=[COLOR_BAND, COLOR_RATIO], edgecolor="black", label="mean")
ax.bar([x + 0.2 for x in xs], maxes, 0.38,
       color=[COLOR_BAND, COLOR_RATIO], edgecolor="black", hatch="//",
       label="max")
ax.axhline(0.0, color="grey", linestyle="--", lw=1.5, label="chance")
ax.set_xticks(xs); ax.set_xticklabels(["Bands (n=24)", "Ratios (n=60)"],
                                      fontsize=11, fontweight="bold")
ax.set_ylabel("R²", fontsize=11, fontweight="bold")
ax.set_title(
    f"B. Weight: bands {bvr['band']['weight_n_sig']}/24 sig, "
    f"ratios {bvr['ratio']['weight_n_sig']}/60 sig",
    fontsize=11, fontweight="bold", loc="left"
)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

fig.suptitle(f"RQ3 — Band vs ratio: which feature type is more informative?   |   {CABLE}",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
out3 = os.path.join(OUT_DIR, f"14_band_vs_ratio_{CABLE}.png")
plt.savefig(out3, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  saved: {out3}")


# ============================================================
# 6. FIGURE 4 — CROSS-TASK CONSISTENCY
# ------------------------------------------------------------
# For each (phase, cell), plot diet bal_acc on x-axis vs weight R^2
# on y-axis. Cells in the upper-right quadrant carry BOTH signals.
# Cells only in the right half carry the diet signal but NOT weight,
# and vice versa. This directly answers RQ3.
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
phase_colors = {"A": "#2C6E9B", "B": "#4C8C4A", "C": "#B7791F", "D": "#8E44AD"}

for phase in ESTROUS_PHASES:
    sub = combined[combined["phase"] == phase].dropna(subset=["bal_acc", "r2"])
    sig_both = (sub["p_diet"] < ALPHA) & (sub["p_weight"] < ALPHA)
    sig_any = ((sub["p_diet"] < ALPHA) | (sub["p_weight"] < ALPHA)) & ~sig_both
    normal = ~(sig_any | sig_both)
    ax.scatter(sub.loc[normal, "bal_acc"], sub.loc[normal, "r2"],
               s=45, alpha=0.5, color=phase_colors[phase],
               edgecolor="black", linewidth=0.4, label=f"phase {phase}")
    ax.scatter(sub.loc[sig_any, "bal_acc"], sub.loc[sig_any, "r2"],
               s=100, alpha=0.9, color=phase_colors[phase],
               edgecolor="#C0392B", linewidth=1.8, marker="o")
    ax.scatter(sub.loc[sig_both, "bal_acc"], sub.loc[sig_both, "r2"],
               s=180, alpha=1.0, color=phase_colors[phase],
               edgecolor="#C0392B", linewidth=2.5, marker="*")

ax.axhline(0, color="grey", linestyle=":", lw=1.0)
ax.axvline(0.5, color="grey", linestyle=":", lw=1.0)
ax.set_xlabel("Diet classification  (balanced accuracy)",
              fontsize=12, fontweight="bold")
ax.set_ylabel("Weight regression  (R²)",
              fontsize=12, fontweight="bold")
ax.set_title(
    f"RQ3 — Cross-task consistency: does a good diet-cell also predict weight?   |   {CABLE}\n"
    "Upper-right quadrant = cells that carry BOTH signals.",
    fontsize=11, fontweight="bold", loc="left"
)

handles = [
    plt.Line2D([], [], marker="o", linestyle="None", markersize=7,
               markerfacecolor="lightgrey", markeredgecolor="black",
               label="n.s."),
    plt.Line2D([], [], marker="o", linestyle="None", markersize=10,
               markerfacecolor="lightgrey", markeredgecolor="#C0392B",
               markeredgewidth=1.8, label="sig. in one task"),
    plt.Line2D([], [], marker="*", linestyle="None", markersize=14,
               markerfacecolor="lightgrey", markeredgecolor="#C0392B",
               markeredgewidth=2.5, label="sig. in BOTH tasks"),
]
leg1 = ax.legend(handles=handles, loc="upper left", fontsize=9)
ax.add_artist(leg1)

# phase legend
phase_handles = [
    plt.Rectangle((0, 0), 1, 1, color=phase_colors[p], label=f"phase {p}")
    for p in ESTROUS_PHASES
]
ax.legend(handles=phase_handles, loc="lower right", fontsize=9,
          title="Phase")
ax.grid(alpha=0.3)
plt.tight_layout()
out4 = os.path.join(OUT_DIR, f"14_cross_task_consistency_{CABLE}.png")
plt.savefig(out4, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  saved: {out4}")


# ============================================================
# 7. FIGURE 5 — GAMMA vs SLOW (delta/theta/beta) BREAKDOWN
# ------------------------------------------------------------
# Only band cells (not ratios); split into slow (delta/theta/beta)
# vs gamma (low_gamma/high_gamma/fast_gamma). Which family carries
# the diet signal?
# ============================================================

sub_bands = combined[combined["cell_type"] == "band"].copy()
sub_bands["family"] = sub_bands["cell"].apply(
    lambda c: "gamma" if c in GAMMA_BANDS else "slow"
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, task_label, metric_col, chance in [
    (axes[0], "Diet",   "bal_acc", 0.5),
    (axes[1], "Weight", "r2",       0.0),
]:
    slow_vals = sub_bands[sub_bands["family"] == "slow"][metric_col].dropna()
    gamma_vals = sub_bands[sub_bands["family"] == "gamma"][metric_col].dropna()
    # violinplot needs at least 2 points per group; fall back to scatter otherwise
    if len(slow_vals) >= 2 and len(gamma_vals) >= 2:
        parts = ax.violinplot([slow_vals, gamma_vals], positions=[0, 1],
                              showmeans=True, showmedians=False)
        for pc, col in zip(parts["bodies"], [COLOR_SLOW, COLOR_GAMMA]):
            pc.set_facecolor(col); pc.set_edgecolor("black"); pc.set_alpha(0.75)
    else:
        # too few points for a violin — plot the individual values instead
        for pos, vals, col in [(0, slow_vals, COLOR_SLOW),
                                (1, gamma_vals, COLOR_GAMMA)]:
            if len(vals):
                ax.scatter([pos] * len(vals), vals, s=80,
                           color=col, edgecolor="black", alpha=0.85)
    ax.axhline(chance, color="grey", linestyle="--", lw=1.5,
               label=f"chance = {chance}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(
        [f"Slow bands\n(delta/theta/beta, n={len(slow_vals)})",
         f"Gamma bands\n(low/high/fast, n={len(gamma_vals)})"],
        fontsize=10, fontweight="bold"
    )
    ax.set_ylabel(metric_col, fontsize=11, fontweight="bold")
    ax.set_title(f"{task_label}", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle(f"RQ3 — Where in the spectrum does the signal live?   |   {CABLE}",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
out5 = os.path.join(OUT_DIR, f"14_gamma_vs_slow_{CABLE}.png")
plt.savefig(out5, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  saved: {out5}")


# ============================================================
# 8. INTERPRETATION TEXT
# ============================================================

# most informative phase per task
best_phase_diet = ph_df.loc[ph_df["diet_max"].idxmax()]
best_phase_wt   = ph_df.loc[ph_df["weight_max"].idxmax()]

# best single cell per task
best_cell_diet = combined.dropna(subset=["bal_acc"]).nlargest(1, "bal_acc").iloc[0]
best_cell_wt   = combined.dropna(subset=["r2"]).nlargest(1, "r2").iloc[0]

# cells significant in BOTH tasks
both_sig = combined[(combined["p_diet"] < ALPHA)
                     & (combined["p_weight"] < ALPHA)]

# gamma vs slow summary
gamma_mean_diet = sub_bands[sub_bands["family"] == "gamma"]["bal_acc"].mean()
slow_mean_diet  = sub_bands[sub_bands["family"] == "slow"]["bal_acc"].mean()
gamma_wins_diet = gamma_mean_diet > slow_mean_diet
gamma_mean_wt = sub_bands[sub_bands["family"] == "gamma"]["r2"].mean()
slow_mean_wt  = sub_bands[sub_bands["family"] == "slow"]["r2"].mean()
gamma_wins_wt = gamma_mean_wt > slow_mean_wt


L = []
sep = "=" * 78
sub_line = "-" * 78
L += [sep, f"14 RQ3 INTERPRETATION — {CABLE}", sep, ""]
L += ["Auto-generated summary of where the diet / weight signals live.", ""]

L += [sub_line, "1. HEADLINE FINDINGS", sub_line]
L += [f"- Diet-signal is significant in {n_sig_diet} / {n_total} cells."]
L += [f"- Weight-signal is significant in {n_sig_weight} / {n_total} cells."]
L += [f"- The best diet cell is phase {best_cell_diet['phase']} x "
      f"{best_cell_diet['cell']} (bal_acc = {best_cell_diet['bal_acc']:.3f})."]
L += [f"- The best weight cell is phase {best_cell_wt['phase']} x "
      f"{best_cell_wt['cell']} (R² = {best_cell_wt['r2']:.3f})."]
L += [""]

L += [sub_line, "2. WHICH ESTROUS PHASE?", sub_line]
L += [f"- Most informative phase for diet   : phase {best_phase_diet['phase']}"]
L += [f"    ({best_phase_diet['diet_n_sig']} significant cells, "
      f"mean bal_acc = {best_phase_diet['diet_mean']:.3f})"]
L += [f"- Most informative phase for weight : phase {best_phase_wt['phase']}"]
L += [f"    ({best_phase_wt['weight_n_sig']} significant cells, "
      f"mean R² = {best_phase_wt['weight_mean']:.3f})"]
L += [""]
L += ["Per-phase table (all cells):"]
for _, r in ph_df.iterrows():
    L.append(f"  phase {r['phase']}: diet {r['diet_n_sig']} sig | "
             f"mean bal_acc {r['diet_mean']:.3f} | max {r['diet_max']:.3f}   "
             f"|| weight {r['weight_n_sig']} sig | "
             f"mean R² {r['weight_mean']:.3f} | max {r['weight_max']:.3f}")
L += [""]

L += [sub_line, "3. BAND vs BAND-TO-BAND RATIO", sub_line]
L += [f"- Bands (n=24):   {bvr['band']['diet_n_sig']} sig diet, "
      f"{bvr['band']['weight_n_sig']} sig weight  |  "
      f"mean bal_acc {bvr['band']['diet_mean']:.3f}, "
      f"mean R² {bvr['band']['weight_mean']:.3f}"]
L += [f"- Ratios (n=60):  {bvr['ratio']['diet_n_sig']} sig diet, "
      f"{bvr['ratio']['weight_n_sig']} sig weight  |  "
      f"mean bal_acc {bvr['ratio']['diet_mean']:.3f}, "
      f"mean R² {bvr['ratio']['weight_mean']:.3f}"]
winner_diet = "ratios" if bvr['ratio']['diet_mean'] > bvr['band']['diet_mean'] else "bands"
winner_wt   = "ratios" if bvr['ratio']['weight_mean'] > bvr['band']['weight_mean'] else "bands"
L += [f"- On average, {winner_diet} carry the stronger diet signal."]
L += [f"- On average, {winner_wt} carry the stronger weight signal."]
L += [""]

L += [sub_line, "4. GAMMA vs SLOW (delta / theta / beta) BAND FAMILY", sub_line]
L += [f"- Diet:   gamma mean bal_acc = {gamma_mean_diet:.3f}, "
      f"slow mean = {slow_mean_diet:.3f}  ->  "
      f"{'gamma wins' if gamma_wins_diet else 'slow wins'}"]
L += [f"- Weight: gamma mean R²      = {gamma_mean_wt:.3f}, "
      f"slow mean = {slow_mean_wt:.3f}  ->  "
      f"{'gamma wins' if gamma_wins_wt else 'slow wins'}"]
L += ["- (Only band cells contribute here; ratios are excluded because they mix bands.)"]
L += [""]

L += [sub_line, "5. CROSS-TASK CONSISTENCY", sub_line]
if len(both_sig) == 0:
    L += ["- No single cell reached p < 0.05 for BOTH diet AND weight simultaneously."]
    L += ["  This is scientifically informative: the diet-signal and the weight-signal"]
    L += ["  do not live in the same cells. LFP encodes diet CATEGORY separately from"]
    L += ["  CONTINUOUS body weight."]
else:
    L += [f"- {len(both_sig)} cells reached p < 0.05 for BOTH diet AND weight:"]
    for _, r in both_sig.iterrows():
        L.append(f"    phase {r['phase']} | {r['cell']:<22s}  "
                 f"bal_acc = {r['bal_acc']:.3f}   R² = {r['r2']:.3f}")
    L += ["  These cells carry a joint diet-and-weight signature — likely the ones"]
    L += ["  most directly linked to metabolic state."]
L += [""]

L += [sub_line, "6. DEFENSE ONE-LINER", sub_line]
L += [f"'The diet signal lives predominantly in phase {best_phase_diet['phase']}, in"]
L += [f"{winner_diet} rather than raw bands, with the best cell being"]
L += [f"{best_cell_diet['cell']} (bal_acc = {best_cell_diet['bal_acc']:.2f}). Weight"]
L += [f"prediction peaks in phase {best_phase_wt['phase']} at {best_cell_wt['cell']}"]
L += [f"(R² = {best_cell_wt['r2']:.2f}). Diet and weight signals do not fully overlap,"]
L += ["which is itself informative: the LFP encodes the two dimensions separately.'"]

out_txt = os.path.join(OUT_DIR, f"14_INTERPRETATION_{CABLE}.txt")
with open(out_txt, "w") as f:
    f.write("\n".join(L))
print(f"  saved: {out_txt}")

# console echo
print("\n" + "=" * 70)
print("QUICK READ")
print("=" * 70)
print(f"  Best diet cell : phase {best_cell_diet['phase']} | "
      f"{best_cell_diet['cell']}  ({best_cell_diet['bal_acc']:.3f})")
print(f"  Best weight cell: phase {best_cell_wt['phase']} | "
      f"{best_cell_wt['cell']}  (R² = {best_cell_wt['r2']:.3f})")
print(f"  Most informative phase (diet)  : {best_phase_diet['phase']}")
print(f"  Most informative phase (weight): {best_phase_wt['phase']}")
print(f"  Bands vs ratios (diet)  : {winner_diet} wins on average")
print(f"  Gamma vs slow (diet)    : {'gamma' if gamma_wins_diet else 'slow'} wins")
print(f"  Cells sig. in both tasks: {len(both_sig)}")
print(f"\nAll outputs saved to:\n  {OUT_DIR}")
print("STEP 14 finished successfully.")
