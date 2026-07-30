# ============================================================
# 04G_PLOT_ALL_RECORDINGS_PSD.PY  (FILTERED VERSION)
# Purpose:
# Plot ALL recording-level PSDs together for visual outlier check.
# Color-coded by group (CTRL = blue, HF = red).
# Each line = one recording.
#
# Normal-scale + log-scale figures.
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

BASE_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

PSD_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    f"04a_multitaper_psd_summary_filtered_{CABLE}.csv"
)

OUTPUT_FOLDER = os.path.join(
    OUTPUT_DIR,
    f"04g_all_recordings_psd_check_filtered_{CABLE}"
)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GROUP_COLORS = {"CTRL": "tab:blue", "HF": "tab:red"}


# ============================================================
# 2. LOAD PSD SUMMARY
# ============================================================

if not os.path.exists(PSD_SUMMARY_PATH):
    raise FileNotFoundError(
        f"PSD summary not found:\n{PSD_SUMMARY_PATH}\n"
        f"Run step 04a (filtered) first."
    )

psd_summary = pd.read_csv(PSD_SUMMARY_PATH)
psd_summary = psd_summary[psd_summary["status"] == "OK"].reset_index(drop=True)

print(f"\nLoaded PSD summary:")
print(f"  Successful recordings: {len(psd_summary)}")
print(psd_summary.groupby("group").size())


# ============================================================
# 3. PLOT HELPER
# ============================================================

def plot_all_recordings(use_log, output_path):
    plt.figure(figsize=(13, 6))

    for _, row in psd_summary.iterrows():
        mouse = row["mouse"]
        group = row["group"]
        psd_path = row["psd_csv_path"]

        if not os.path.exists(psd_path):
            print(f"PSD not found, skipping: {psd_path}")
            continue

        psd_df = pd.read_csv(psd_path)
        color = GROUP_COLORS.get(group, "gray")

        plt.plot(
            psd_df["frequency_Hz"],
            psd_df["mean_power"],
            linewidth=0.7,
            color=color,
            alpha=0.55,
        )

    # Legend just for groups, not each recording
    for g, c in GROUP_COLORS.items():
        n = (psd_summary["group"] == g).sum()
        plt.plot([], [], color=c, label=f"{g} (n={n} recordings)")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power" + (" (log scale)" if use_log else ""))
    if use_log:
        plt.yscale("log")

    title = f"All Recording-level PSDs ({CABLE})"
    if use_log:
        title += " — log scale"
    plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"\nSaved: {output_path}")


# ============================================================
# 4. NORMAL + LOG SCALE
# ============================================================

plot_all_recordings(
    use_log=False,
    output_path=os.path.join(
        OUTPUT_FOLDER,
        f"04g_all_recordings_psd_{CABLE}.png"
    )
)

plot_all_recordings(
    use_log=True,
    output_path=os.path.join(
        OUTPUT_FOLDER,
        f"04g_all_recordings_psd_logscale_{CABLE}.png"
    )
)

print("\nSTEP 04G (filtered) finished successfully.")

