# ============================================================
# PLOT_PSD_BANDS.PY
# Purpose:
# Produce a publication-quality figure of a single multitaper PSD
# with the six analysis bands shaded, and the 10-15 Hz gap marked.
#
# Reads one PSD CSV (from 04a, columns: frequency_Hz, mean_power).
# Writes one PNG (log-log, band-annotated).
#
# Usage:
#   1. Set PSD_CSV_PATH below to the recording you want to plot.
#   2. (Optional) set OUTPUT_PNG_PATH if you want a specific location.
#   3. Run:  python plot_psd_bands.py
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. USER SETTINGS  ->  edit these
# ============================================================

# --- input: which recording's PSD to plot ---
PSD_CSV_PATH = (
    "/Users/amanlizahra/Desktop/thesis_lfp_analysis/outputs/"
    "04a_multitaper_psd_filtered_Cable1/csv/"
    "Mouse1_Cable1_HF_2026-03-19_12-55-20_"
    "DIFF_Ch2_minus_Ch3_multitaper_psd.csv"
)

# --- output: where to save the PNG ---
# By default the PNG is written to the SAME folder as the PSD CSV,
# so it appears next to the input file in your outputs tree.
#
# To send all figures to a single "thesis figures" folder instead,
# set OUTPUT_DIR to that folder, e.g.:
#     OUTPUT_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis/outputs/thesis_figures"
# The folder will be created automatically if it does not exist.
OUTPUT_DIR = None    # None = save next to the CSV

# --- optional caption info shown in a small box on the plot ---
# Leave any field as None to hide it. All optional.
INFO = {
    "mouse":         None,   # e.g. 1
    "group":         None,   # e.g. "HF"
    "days_on_diet":  None,   # e.g. 24
    "body_weight_g": None,   # e.g. 35.7
}

# --- figure size / DPI ---
FIG_WIDTH  = 8.4     # inches
FIG_HEIGHT = 4.8
DPI        = 300


# ============================================================
# 2. ANALYSIS BANDS  ->  keep in sync with 05a
# ------------------------------------------------------------
# (name, low Hz, high Hz, fill colour, text colour)
# ============================================================

BANDS = [
    ("delta",       1,   4,   "#F4E4E1", "#8C4A3D"),
    ("theta",       4,   10,  "#F1EAD8", "#7A5A0F"),
    ("beta",        15,  30,  "#E2ECF6", "#1E4F86"),
    ("low\ngamma",  30,  60,  "#DDEEE7", "#0F6E56"),
    ("high\ngamma", 60,  100, "#E4E1F3", "#3D3488"),
    ("fast\ngamma", 100, 140, "#F0DDEB", "#7A2A5F"),
]

# The 10-15 Hz range is deliberately not assigned to any band;
# it is left as a visible blank strip between theta and beta.
FREQ_MIN, FREQ_MAX = 1, 140


# ============================================================
# 3. LOAD
# ============================================================

if not os.path.exists(PSD_CSV_PATH):
    raise FileNotFoundError(f"PSD CSV not found:\n{PSD_CSV_PATH}")

psd = pd.read_csv(PSD_CSV_PATH)
if not {"frequency_Hz", "mean_power"}.issubset(psd.columns):
    raise ValueError(
        "PSD CSV must contain columns 'frequency_Hz' and 'mean_power'."
    )

f = psd["frequency_Hz"].values
p = psd["mean_power"].values

# clip to the plotted range (log axes hate zero / negative)
mask = (f >= FREQ_MIN) & (f <= FREQ_MAX) & (p > 0)
f, p = f[mask], p[mask]


# ============================================================
# 4. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ---- band shading (behind the curve) ----
for name, lo, hi, fc, tc in BANDS:
    ax.axvspan(lo, hi, color=fc, alpha=0.85, zorder=0)

# ---- PSD curve ----
ax.plot(f, p, color="#1A1A18", linewidth=1.4, zorder=3)

# ---- log-log axes ----
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(FREQ_MIN, FREQ_MAX)
ax.set_ylim(p.min() * 0.7, p.max() * 2.2)

# ---- axis labels & ticks ----
ax.set_xlabel("Frequency (Hz)", fontsize=11, color="#2C2C2A")
ax.set_ylabel("Power spectral density (a.u.)",
              fontsize=11, color="#2C2C2A")

xticks = [1, 4, 10, 15, 30, 60, 100, 140]
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks],
                   fontsize=9.5, color="#2C2C2A")
ax.tick_params(axis="y", labelsize=9.5, colors="#5F5E5A")

# clean spines
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#B4B2A9")
ax.spines["bottom"].set_color("#B4B2A9")

# ---- band name labels ----
# Theta label sits slightly to the left to avoid the theta peak;
# delta label sits lower to avoid the optional info box in the
# top-left corner; all other labels sit near the top of the axes.
ymax = ax.get_ylim()[1]
label_y_top   = ymax * 0.42
label_y_delta = ymax * 0.10
for name, lo, hi, fc, tc in BANDS:
    if name == "theta":
        x = 4.6                       # left side of theta band, away from peak
    else:
        x = np.sqrt(lo * hi)          # geometric centre on log x-axis
    y = label_y_delta if name == "delta" else label_y_top
    ax.text(x, y, name,
            ha="center", va="center",
            fontsize=8.2, color=tc,
            fontweight="bold", linespacing=0.95)

# ---- optional info box (top-left) ----
info_lines = []
if INFO.get("mouse") is not None:
    info_lines.append(f"Mouse {INFO['mouse']}")
if INFO.get("group"):
    info_lines.append(f"{INFO['group']}")
if INFO.get("days_on_diet") is not None:
    info_lines.append(f"day {INFO['days_on_diet']} on diet")
if INFO.get("body_weight_g") is not None:
    info_lines.append(f"{INFO['body_weight_g']:.1f} g")

if info_lines:
    ax.text(0.015, 0.965, "\n".join(info_lines),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8.5, color="#2C2C2A",
            bbox=dict(facecolor="white",
                      edgecolor="#B4B2A9",
                      boxstyle="round,pad=0.35",
                      linewidth=0.6))

plt.tight_layout()


# ============================================================
# 5. SAVE
# ============================================================

csv_base = os.path.splitext(os.path.basename(PSD_CSV_PATH))[0]
png_name = f"{csv_base}_bands.png"

if OUTPUT_DIR is None:
    # Save next to the CSV
    output_png_path = os.path.join(os.path.dirname(PSD_CSV_PATH), png_name)
else:
    # Save into the user-chosen folder (create it if needed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_png_path = os.path.join(OUTPUT_DIR, png_name)

plt.savefig(output_png_path,
            dpi=DPI, facecolor="white",
            bbox_inches="tight", pad_inches=0.15)
plt.close()

print(f"Saved figure to:\n{output_png_path}")