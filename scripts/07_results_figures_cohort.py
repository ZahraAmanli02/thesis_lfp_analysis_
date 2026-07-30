# ============================================================
# 07_RESULTS_FIGURES_COHORT.PY
# Purpose:
# Generate thesis-ready cohort figures for Section 4.1:
#   Figure 4.2 — Recordings per mouse (single cable)
#   Figure 4.3 — Estrous phase distribution across groups
#                (single cable, with chi-square in title)
#
# Runs ONE cable at a time (set CABLE below). Run once with
# CABLE = "Cable1", then again with CABLE = "Cable3".
# Each run produces its own PNG + PDF pair per figure.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#
# Output:
#   outputs/07_results_figures_<CABLE>/
#       Figure_4_2_recordings_per_mouse_<CABLE>.{png,pdf}
#       Figure_4_3_estrous_distribution_<CABLE>.{png,pdf}
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BAND_POWERS_PATH = os.path.join(
    OUTPUT_DIR, f"05a_band_powers_{CABLE}", f"05a_band_powers_{CABLE}.csv"
)

OUT_DIR = os.path.join(OUTPUT_DIR, f"07_results_figures_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)

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
# 2. LOAD DATA
# ============================================================

if not os.path.exists(BAND_POWERS_PATH):
    raise FileNotFoundError(f"Missing 05a output for {CABLE}:\n{BAND_POWERS_PATH}")

df = pd.read_csv(BAND_POWERS_PATH)
print(f"Loaded {CABLE}: {len(df)} recordings, {df['mouse'].nunique()} mice")


# ============================================================
# 3. FIGURE 4.2 — Recordings per mouse
# ============================================================

fig, ax = plt.subplots(figsize=(6, 4))

# Order: CTRL mice first (by ID), then HF mice (by ID)
per = df.groupby(["mouse", "group"]).size().reset_index(name="n")
ctrl = per[per["group"] == "CTRL"].sort_values("mouse")
hf   = per[per["group"] == "HF"].sort_values("mouse")
ordered = pd.concat([ctrl, hf], ignore_index=True)

xpos = np.arange(len(ordered))
colors = [C_CTRL if g == "CTRL" else C_HF for g in ordered["group"]]

ax.bar(xpos, ordered["n"], color=colors, alpha=0.85,
       edgecolor="black", linewidth=0.4)
ax.set_xticks(xpos)
ax.set_xticklabels([f"M{m}" for m in ordered["mouse"]],
                   rotation=0, fontsize=8)
ax.set_ylabel("Recordings")
ax.set_title(CABLE, fontweight="medium", fontsize=11)
ax.set_ylim(0, ordered["n"].max() + 1)

# Legend
ax.bar([-99], [0], color=C_CTRL, alpha=0.85,
       label=f"CTRL ({len(ctrl)} mice)")
ax.bar([-99], [0], color=C_HF,   alpha=0.85,
       label=f"HF ({len(hf)} mice)")
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_xlim(-0.6, len(ordered) - 0.4)

plt.tight_layout()
fig_42_png = os.path.join(OUT_DIR, f"Figure_4_2_recordings_per_mouse_{CABLE}.png")
fig_42_pdf = os.path.join(OUT_DIR, f"Figure_4_2_recordings_per_mouse_{CABLE}.pdf")
plt.savefig(fig_42_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(fig_42_pdf,           bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {fig_42_png}")
print(f"Saved: {fig_42_pdf}")


# ============================================================
# 4. FIGURE 4.3 — Estrous phase distribution + chi-square
# ============================================================

phases = ["A", "B", "C", "D"]
x = np.arange(len(phases))
width = 0.4

fig, ax = plt.subplots(figsize=(5, 4))

ct = pd.crosstab(df["estrous_phase"], df["group"]).reindex(phases).fillna(0)
chi2, p, dof, _ = stats.chi2_contingency(ct)
ctrl_counts = ct["CTRL"].values
hf_counts   = ct["HF"].values

ax.bar(x - width/2, ctrl_counts, width, color=C_CTRL, alpha=0.85,
       edgecolor="black", linewidth=0.4, label="CTRL")
ax.bar(x + width/2, hf_counts,   width, color=C_HF,   alpha=0.85,
       edgecolor="black", linewidth=0.4, label="HF")

# Counts above bars
for i, c in enumerate(ctrl_counts):
    ax.text(i - width/2, c + 0.5, f"{int(c)}",
            ha="center", fontsize=8)
for i, c in enumerate(hf_counts):
    ax.text(i + width/2, c + 0.5, f"{int(c)}",
            ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(phases)
ax.set_xlabel("Estrous phase")
ax.set_ylabel("Recordings")

p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
ax.set_title(f"{CABLE}   (chi2({dof}) = {chi2:.2f}, {p_str})",
             fontweight="medium", fontsize=10)
ax.set_ylim(0, max(ct.values.max(), 1) + 4)
ax.legend(loc="upper left", frameon=False, fontsize=9)

plt.tight_layout()
fig_43_png = os.path.join(OUT_DIR, f"Figure_4_3_estrous_distribution_{CABLE}.png")
fig_43_pdf = os.path.join(OUT_DIR, f"Figure_4_3_estrous_distribution_{CABLE}.pdf")
plt.savefig(fig_43_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(fig_43_pdf,           bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {fig_43_png}")
print(f"Saved: {fig_43_pdf}")

print("\nDone.")