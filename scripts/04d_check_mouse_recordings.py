# ============================================================
# 04D_CHECK_MOUSE_RECORDINGS.PY 
# Purpose:
# Check all recording-level PSDs for one selected mouse.
# Used to find which specific recording causes outlier behavior
# seen in 04d.
#
# Change TARGET_MOUSE below to inspect different mice.
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

CABLE = "Cable1"

# Which mouse to inspect — change as needed
TARGET_MOUSE = 26

BASE_DIR = "/Users/amanlizahra/Desktop/thesis_lfp_analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

PSD_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    f"04a_multitaper_psd_summary_filtered_{CABLE}.csv"
)

OUTPUT_FOLDER = os.path.join(
    OUTPUT_DIR,
    f"04d_mouse{TARGET_MOUSE}_recording_psd_check_filtered_{CABLE}"
)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD PSD SUMMARY
# ============================================================

if not os.path.exists(PSD_SUMMARY_PATH):
    raise FileNotFoundError(
        f"PSD summary not found:\n{PSD_SUMMARY_PATH}"
    )

psd_summary = pd.read_csv(PSD_SUMMARY_PATH)


# ============================================================
# 3. SELECT TARGET MOUSE
# ============================================================

mouse_df = psd_summary[
    (psd_summary["mouse"] == TARGET_MOUSE) &
    (psd_summary["status"] == "OK")
].reset_index(drop=True)

print(f"\nMouse {TARGET_MOUSE} recordings:")
print(mouse_df[["mouse", "group", "differential_file",
                "days_on_diet", "body_weight"]])
print(f"Number of recordings: {len(mouse_df)}")

if len(mouse_df) == 0: 
    raise ValueError(
        f"No successful PSD recordings for Mouse {TARGET_MOUSE}."
    )


# ============================================================
# 4. SHORT LABEL HELPER
# ============================================================

def make_label(row):
    """Compact label: e.g. '03-16 / d14 / 28.5g' (date / days_on_diet / weight)."""
    # Short date MM-DD (year is always 2026, so it is dropped)
    date_str = ""
    raw_date = row.get("recording_date", "")
    if pd.notna(raw_date) and str(raw_date).strip():
        parsed = pd.to_datetime(raw_date, errors="coerce")
        if pd.notna(parsed):
            date_str = parsed.strftime("%m-%d")

    days = row.get("days_on_diet", "")
    weight = row.get("body_weight", "")
    if pd.notna(days):
        days = f"d{int(days)}"
    else:
        days = ""
    if pd.notna(weight):
        weight = f"{weight:.1f}g"
    else:
        weight = ""
    parts = [s for s in [date_str, days, weight] if s]
    return " / ".join(parts) if parts else row["differential_file"][:25]


# ============================================================
# 5. PLOT ALL RECORDING-LEVEL PSDs (NORMAL SCALE)
# ============================================================

plt.figure(figsize=(14, 7))
peak_rows = []

for _, row in mouse_df.iterrows():
    psd_path = row["psd_csv_path"]
    diff_file = row["differential_file"]

    if not os.path.exists(psd_path):
        print(f"  ⚠️  PSD not found: {psd_path}")
        continue

    psd_df = pd.read_csv(psd_path)
    frequency = psd_df["frequency_Hz"]
    power = psd_df["mean_power"]

    max_power = power.max()
    max_freq = frequency[power.idxmax()]

    label = make_label(row)

    peak_rows.append({
        "recording": diff_file,
        "label": label,
        "days_on_diet": row.get("days_on_diet", ""),
        "body_weight": row.get("body_weight", ""),
        "max_power": max_power,
        "max_power_frequency_Hz": max_freq,
        "psd_csv_path": psd_path,
    })

    plt.plot(frequency, power, linewidth=1, label=label)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title(f"Mouse {TARGET_MOUSE} Recording-level PSDs ({CABLE})")
plt.legend(fontsize=8)
plt.tight_layout()

fig_path = os.path.join(
    OUTPUT_FOLDER,
    f"04d_Mouse{TARGET_MOUSE}_all_recording_psds_{CABLE}.png"
)
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"\nSaved: {fig_path}")


# ============================================================
# 6. PEAK POWER TABLE
# ============================================================

peak_df = pd.DataFrame(peak_rows)
peak_df = peak_df.sort_values(by="max_power", ascending=False).reset_index(drop=True)

peak_table_path = os.path.join(
    OUTPUT_FOLDER,
    f"04d_Mouse{TARGET_MOUSE}_recording_peak_power_table_{CABLE}.csv"
)
peak_df.to_csv(peak_table_path, index=False)

print("\nTop recordings by peak power:")
print(peak_df[["label", "days_on_diet", "body_weight",
               "max_power", "max_power_frequency_Hz"]].head(10).to_string(index=False))


# ============================================================
# 7. LOG-SCALE PLOT
# ============================================================

plt.figure(figsize=(14, 7))

for _, row in mouse_df.iterrows():
    psd_path = row["psd_csv_path"]
    if not os.path.exists(psd_path):
        continue

    psd_df = pd.read_csv(psd_path)
    plt.plot(
        psd_df["frequency_Hz"],
        psd_df["mean_power"],
        linewidth=1,
        label=make_label(row),
    )

plt.yscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (log)")
plt.title(
    f"Mouse {TARGET_MOUSE} Recording-level PSDs (log scale) ({CABLE})"
)
plt.legend(fontsize=8)
plt.tight_layout()

log_fig_path = os.path.join(
    OUTPUT_FOLDER,
    f"04d_Mouse{TARGET_MOUSE}_all_recording_psds_logscale_{CABLE}.png"
)
plt.savefig(log_fig_path, dpi=300)
plt.show()
print(f"\nSaved: {log_fig_path}")


print("\nSTEP 04D (filtered) finished successfully.")