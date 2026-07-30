# ============================================================
# 07_GROUP_PSD_FIGURE.PY
# Purpose:
# Generate Figure 4.5 (Section 4.3): group-average power spectral
# density curves for the analytic sample of one cable.
#
# Pipeline:
#   1. For each recording in the clean set (per 05a), load its
#      per-recording multitaper PSD CSV (produced by 04a).
#   2. Normalize each PSD to relative power (integral over
#      1–140 Hz = 1), matching the "fraction of 1–140 Hz"
#      convention used in the existing 05ab feature overview.
#   3. Compute mouse-level mean PSD (average across recordings
#      within each mouse). This is the correct aggregation
#      level given the nested structure (Section 4.1.2).
#   4. Compute group-level mean and SEM across mice.
#   5. Plot on log-log axes; save PNG + PDF + summary CSV.
#
# Runs one cable at a time.
#
# Input:
#   outputs/05a_band_powers_<CABLE>/05a_band_powers_<CABLE>.csv
#   outputs/04a_multitaper_psd_<CABLE>/*.csv
#       (per-recording PSDs; two columns: frequency_Hz, mean_power)
#
# Output:
#   outputs/07_group_psd_figure_<CABLE>/
#       Figure_4_5_group_psd_<CABLE>.{png,pdf}
#       07_group_psd_summary_<CABLE>.csv
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def _resolve_05a_path(output_dir, cable):
    candidates = [
        os.path.join(output_dir, f"05a_band_powers_{cable}",
                     f"05a_band_powers_{cable}.csv"),
        os.path.join(output_dir, f"05a_band_powers_{cable}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Missing 05a output for {cable}. Tried:\n  " + "\n  ".join(candidates)
    )

BAND_POWERS_PATH = _resolve_05a_path(OUTPUT_DIR, CABLE)

# Folder holding per-recording PSD CSVs.
# Tries the known "04a_multitaper_psd_filtered_<CABLE>/csv/" pattern first,
# then falls back to simpler variants.
def _resolve_psd_dir(output_dir, cable):
    candidates = [
        os.path.join(output_dir, f"04a_multitaper_psd_filtered_{cable}", "csv"),
        os.path.join(output_dir, f"04a_multitaper_psd_{cable}", "csv"),
        os.path.join(output_dir, f"04a_multitaper_psd_filtered_{cable}"),
        os.path.join(output_dir, f"04a_multitaper_psd_{cable}"),
        os.path.join(output_dir, "04a_multitaper_psd"),
    ]
    for p in candidates:
        if os.path.isdir(p) and glob.glob(os.path.join(p, "*_multitaper_psd.csv")):
            return p
    raise FileNotFoundError(
        f"Cannot find PSD folder for {cable}. Tried:\n  " + "\n  ".join(candidates)
    )

PSD_DIR = _resolve_psd_dir(OUTPUT_DIR, CABLE)

OUT_DIR = os.path.join(OUTPUT_DIR, f"07_group_psd_figure_{CABLE}")
os.makedirs(OUT_DIR, exist_ok=True)

# Integration bounds (must match 04a band-power scheme)
FREQ_MIN = 1.0
FREQ_MAX = 140.0

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
# 2. LOAD CLEAN SET AND BUILD LOOKUP
# ------------------------------------------------------------
# 05a lists the clean recordings for this cable. Its
# differential_file column matches the base of each per-recording
# PSD filename (strip the "_multitaper_psd.csv" suffix).
# ============================================================

df05a = pd.read_csv(BAND_POWERS_PATH)
print(f"Clean set from 05a ({CABLE}): {len(df05a)} recordings, "
      f"{df05a['mouse'].nunique()} mice")

# Build a set of expected filename stems.
# NOTE: 05a stores differential_file with a .npy extension
# (e.g. "MouseN_..._DIFF_ChX_minus_ChY.npy") while the per-recording
# PSD files are named "MouseN_..._DIFF_ChX_minus_ChY_multitaper_psd.csv"
# — so we strip the .npy suffix before comparing.
def _stem_from_diff_file(s):
    s = str(s)
    return s[:-4] if s.endswith(".npy") else s

df05a["_stem"] = df05a["differential_file"].apply(_stem_from_diff_file)
expected_stems = set(df05a["_stem"].values)
lookup = df05a.set_index("_stem")[["mouse", "group"]]


# ============================================================
# 3. LOAD AND NORMALIZE PER-RECORDING PSDS
# ============================================================

psd_files = sorted(glob.glob(os.path.join(PSD_DIR, "*_multitaper_psd.csv")))
print(f"Found {len(psd_files)} PSD files in {PSD_DIR}")

# Common frequency grid (assume all files use the same one)
first = pd.read_csv(psd_files[0])
freqs = first["frequency_Hz"].values
mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
freqs = freqs[mask]

records = []   # rows: dict(mouse, group, rel_power[len(freqs)])
skipped_not_in_clean = 0
skipped_missing = 0

for path in psd_files:
    stem = os.path.basename(path).replace("_multitaper_psd.csv", "")
    if stem not in expected_stems:
        skipped_not_in_clean += 1
        continue
    df = pd.read_csv(path)
    if len(df) != len(mask):
        # frequency grid mismatch — align by mask
        df = df[df["frequency_Hz"].between(FREQ_MIN, FREQ_MAX)].reset_index(drop=True)
    else:
        df = df[mask].reset_index(drop=True)
    power = df["mean_power"].values
    if np.any(~np.isfinite(power)) or power.sum() <= 0:
        skipped_missing += 1
        continue
    # Normalize to relative power (integral = 1)
    df_freq = df["frequency_Hz"].values
    integ = np.trapz(power, df_freq)
    rel = power / integ if integ > 0 else power * 0
    row = lookup.loc[stem]
    records.append({
        "mouse": int(row["mouse"]),
        "group": row["group"],
        "rel_power": rel,
    })

print(f"Retained {len(records)} clean recordings; "
      f"skipped {skipped_not_in_clean} (not in clean set), "
      f"{skipped_missing} (bad values)")

if len(records) == 0:
    raise RuntimeError("No PSDs matched the clean set — check file naming.")

# Stack into matrix: rows = recordings, cols = freq
rel_matrix = np.vstack([r["rel_power"] for r in records])
meta = pd.DataFrame([{"mouse": r["mouse"], "group": r["group"]} for r in records])


# ============================================================
# 4. MOUSE-LEVEL AND GROUP-LEVEL AGGREGATION
# ============================================================

# Mouse-level: mean rel_power across recordings within each mouse
mouse_means = {}   # (mouse, group) -> array
for (mouse, group), sub in meta.groupby(["mouse", "group"]):
    idx = sub.index.values
    mouse_means[(mouse, group)] = rel_matrix[idx].mean(axis=0)

# Group-level: mean & SEM across mouse means
groups_data = {}
for grp in ["CTRL", "HF"]:
    stacks = [v for (m, g), v in mouse_means.items() if g == grp]
    if len(stacks) == 0:
        continue
    arr = np.vstack(stacks)
    groups_data[grp] = {
        "n_mice": arr.shape[0],
        "mean":   arr.mean(axis=0),
        "sem":    arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]),
    }
    print(f"  {grp}: {arr.shape[0]} mice")


# ============================================================
# 5. FIGURE 4.5
# ============================================================

fig, ax = plt.subplots(figsize=(6.5, 4.5))

for grp, col in [("CTRL", C_CTRL), ("HF", C_HF)]:
    if grp not in groups_data:
        continue
    d = groups_data[grp]
    ax.plot(freqs, d["mean"], color=col, lw=2,
            label=f"{grp} (n = {d['n_mice']} mice)")
    ax.fill_between(freqs, d["mean"] - d["sem"], d["mean"] + d["sem"],
                    color=col, alpha=0.25)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Relative power (fraction of 1–140 Hz)")
ax.set_title(f"Group-average LH PSD — {CABLE}", fontsize=11)
ax.legend(loc="lower left", frameon=False, fontsize=9)
ax.grid(True, which="both", alpha=0.15, lw=0.5)

# Frequency-band shading (optional visual guide)
band_edges = [(1, 4, "δ"), (4, 10, "θ"), (15, 30, "β"),
              (30, 60, "low γ"), (60, 100, "high γ"), (100, 140, "fast γ")]
for lo, hi, lab in band_edges:
    ax.axvspan(lo, hi, alpha=0.03, color="grey", zorder=0)

plt.tight_layout()
fig_png = os.path.join(OUT_DIR, f"Figure_4_5_group_psd_{CABLE}.png")
fig_pdf = os.path.join(OUT_DIR, f"Figure_4_5_group_psd_{CABLE}.pdf")
plt.savefig(fig_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(fig_pdf,           bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {fig_png}")


# ============================================================
# 6. SUMMARY CSV (for thesis text and later re-plotting)
# ============================================================

summary = pd.DataFrame({"frequency_Hz": freqs})
for grp in ["CTRL", "HF"]:
    if grp in groups_data:
        summary[f"{grp}_mean"] = groups_data[grp]["mean"]
        summary[f"{grp}_sem"]  = groups_data[grp]["sem"]

summary_path = os.path.join(OUT_DIR, f"07_group_psd_summary_{CABLE}.csv")
summary.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")
print("\nDone.")