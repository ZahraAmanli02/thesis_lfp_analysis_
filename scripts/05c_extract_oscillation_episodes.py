# ============================================================
# 05C_EXTRACT_OSCILLATION_EPISODES.PY  (RECORDING-LEVEL)
# Purpose:
# Detect discrete oscillation episodes (bursts) in beta and the two
# gamma sub-bands of each differential recording. Band power (05a)
# and ratios (05b) describe "how much energy on average"; episodes
# describe "how that energy is distributed in time" — how often the
# rhythm bursts, how long each burst lasts, how strong it is.
#
# Bands (match the 05a/05b band scheme):
#   beta        15-30 Hz
#   low_gamma   30-60 Hz   (slow-gamma / feeding assemblies, Altafi 2024)
#   high_gamma  60-100 Hz  (high-gamma / food-seeking, Carus-Cadavieco 2017)
#
# fast_gamma (100-140 Hz) is intentionally NOT included here: episode
# detection needs a clean envelope, and that band sits close to the
# EMG / noise floor where the envelope is unreliable.
#
# Method (per band):
#   1) Band-pass filter the signal (zero-phase, 4th-order Butterworth)
#   2) Compute the Hilbert envelope (instantaneous amplitude)
#   3) Threshold = median + N_MAD * MAD of the envelope (robust)
#   4) An "episode" = contiguous region above threshold, long enough
#      (>= MIN_EPISODE_DURATION_SEC); near-touching episodes (gap <
#      MAX_GAP_SEC) are merged so one burst is not split in two.
#
# Per band, extract:
#   - episode_rate         (episodes per minute)
#   - mean_duration_sec    (average episode length)
#   - mean_amplitude       (average envelope during episodes)
#   - fraction_of_time     (fraction of recording spent in episodes)
#   - n_episodes, threshold (bookkeeping)
#
# Like 05a/05b, confirmed-bad recordings (from 04b) are excluded
# BEFORE features are computed, so the output contains clean
# recordings only — no separate "cleaned" file is produced.
#
# Input:
#   outputs/03a_differential_summary_filtered_<CABLE>.csv   (signal paths)
#   outputs/04b_bad_recording_check_filtered_<CABLE>/
#       04b_confirmed_bad_recordings_<CABLE>.csv             (exclusion list)
#   outputs/05b_band_ratios_<CABLE>/
#       05b_band_ratios_<CABLE>.csv                          (for the merge)
#
# Output:
#   outputs/05c_oscillation_episodes_<CABLE>/
#       05c_oscillation_episode_features_<CABLE>.csv
#       05c_features_combined_with_episodes_<CABLE>.csv
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, hilbert


# ============================================================
# 1. SETTINGS
# ============================================================

FS = 1250
CABLE = "Cable1"

# Resolve project root from this script's location (scripts/ -> root),
# so paths work regardless of the project folder name.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DIFF_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    f"03a_differential_summary_filtered_{CABLE}.csv"
)

CONFIRMED_BAD_PATH = os.path.join(
    OUTPUT_DIR,
    f"04b_bad_recording_check_filtered_{CABLE}",
    f"04b_confirmed_bad_recordings_{CABLE}.csv"
)

# 05b band-ratio table (clean recordings only) — used for the merge
BAND_RATIOS_PATH = os.path.join(
    OUTPUT_DIR,
    f"05b_band_ratios_{CABLE}",
    f"05b_band_ratios_{CABLE}.csv"
)

OUTPUT_DIR_05C = os.path.join(
    OUTPUT_DIR,
    f"05c_oscillation_episodes_{CABLE}"
)
os.makedirs(OUTPUT_DIR_05C, exist_ok=True)

EPISODE_FEATURES_PATH = os.path.join(
    OUTPUT_DIR_05C,
    f"05c_oscillation_episode_features_{CABLE}.csv"
)

FINAL_COMBINED_PATH = os.path.join(
    OUTPUT_DIR_05C,
    f"05c_features_combined_with_episodes_{CABLE}.csv"
)


# ============================================================
# 2. DETECTION PARAMETERS
# ============================================================

# Bands match the 05a/05b scheme. fast_gamma excluded (see header).
BANDS = {
    "beta":        (15, 30),
    "low_gamma":   (30, 60),
    "high_gamma":  (60, 100),
}

FILTER_ORDER = 4

# Threshold = median + N_MAD * MAD of the envelope.
# Higher value -> fewer, stronger episodes detected.
N_MAD = 3.0

# Minimum episode duration (shorter ones are ignored).
MIN_EPISODE_DURATION_SEC = 0.1

# Gap-bridging: episodes closer than this are merged, so a single
# burst with a brief dip is not counted as two episodes.
MAX_GAP_SEC = 0.05


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = fs / 2
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)


def detect_episodes(envelope, fs, threshold,
                    min_duration_sec, max_gap_sec):
    """
    Detect episodes where envelope > threshold.
    Returns a list of (start_idx, end_idx) tuples.
    """
    above = envelope > threshold

    # Run starts and ends
    transitions = np.diff(above.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    # Signal already above threshold at the very start / end
    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, len(above))

    # Pair starts and ends defensively
    n = min(len(starts), len(ends))
    starts = starts[:n]
    ends = ends[:n]

    raw_episodes = list(zip(starts, ends))
    if len(raw_episodes) == 0:
        return []

    # Merge episodes that are very close together
    max_gap_samples = int(max_gap_sec * fs)
    merged = [raw_episodes[0]]
    for s, e in raw_episodes[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= max_gap_samples:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    # Discard episodes shorter than the minimum
    min_samples = int(min_duration_sec * fs)
    final = [(s, e) for s, e in merged if (e - s) >= min_samples]
    return final


def compute_episode_features(signal, fs, band_name, lowcut, highcut):
    """Return the episode features for one band as a dict."""
    # 1) Band-pass filter
    filtered = bandpass_filter(signal, fs, lowcut, highcut, FILTER_ORDER)

    # 2) Hilbert envelope (instantaneous amplitude)
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    # 3) Robust threshold
    median = np.median(envelope)
    mad = np.median(np.abs(envelope - median))
    if mad == 0:
        mad = np.std(envelope)  # fall back if MAD is 0
    threshold = median + N_MAD * mad

    # 4) Detect episodes
    episodes = detect_episodes(
        envelope=envelope,
        fs=fs,
        threshold=threshold,
        min_duration_sec=MIN_EPISODE_DURATION_SEC,
        max_gap_sec=MAX_GAP_SEC,
    )

    total_duration_sec = len(signal) / fs

    if len(episodes) == 0:
        return {
            f"{band_name}_episode_rate":      0.0,
            f"{band_name}_mean_duration_sec": np.nan,
            f"{band_name}_mean_amplitude":    np.nan,
            f"{band_name}_fraction_of_time":  0.0,
            f"{band_name}_n_episodes":        0,
            f"{band_name}_threshold":         float(threshold),
        }

    durations_samples = np.array([e - s for s, e in episodes])
    durations_sec = durations_samples / fs
    amplitudes = np.array([np.mean(envelope[s:e]) for s, e in episodes])

    n_episodes = len(episodes)
    rate_per_min = n_episodes / (total_duration_sec / 60)
    mean_duration_sec = float(np.mean(durations_sec))
    mean_amplitude = float(np.mean(amplitudes))
    fraction_of_time = float(np.sum(durations_samples) / len(signal))

    return {
        f"{band_name}_episode_rate":      rate_per_min,
        f"{band_name}_mean_duration_sec": mean_duration_sec,
        f"{band_name}_mean_amplitude":    mean_amplitude,
        f"{band_name}_fraction_of_time":  fraction_of_time,
        f"{band_name}_n_episodes":        n_episodes,
        f"{band_name}_threshold":         float(threshold),
    }


# ============================================================
# 4. LOAD DIFFERENTIAL SUMMARY (need the signal paths)
# ============================================================

if not os.path.exists(DIFF_SUMMARY_PATH):
    raise FileNotFoundError(
        f"Differential summary not found:\n{DIFF_SUMMARY_PATH}\n"
        f"Run step 03a first."
    )

summary_df = pd.read_csv(DIFF_SUMMARY_PATH)
print(f"\nLoaded differential summary: {len(summary_df)} recordings")

# Keep only successful recordings if a status column exists
if "status" in summary_df.columns:
    summary_df = summary_df[summary_df["status"] == "OK"].reset_index(drop=True)
    print(f"  After status==OK filter: {len(summary_df)} recordings")

if "diff_npy_path" in summary_df.columns:
    path_column = "diff_npy_path"
elif "diff_eeg_path" in summary_df.columns:
    path_column = "diff_eeg_path"
else:
    raise ValueError("Summary missing diff_npy_path / diff_eeg_path.")


# ============================================================
# 5. EXCLUDE CONFIRMED-BAD RECORDINGS (from 04b)
# ------------------------------------------------------------
# Same principle as 05a/05b: remove individual bad RECORDINGS
# (not whole mice) before any feature is computed, so the output
# table holds clean recordings only.
# ============================================================

n_before = len(summary_df)
if os.path.exists(CONFIRMED_BAD_PATH):
    confirmed_bad = pd.read_csv(CONFIRMED_BAD_PATH)
    bad_files = set(
        confirmed_bad.get("differential_file", pd.Series([], dtype=str))
    )
    if bad_files:
        summary_df = summary_df[
            ~summary_df["differential_file"].isin(bad_files)
        ].reset_index(drop=True)
    print(f"Confirmed-bad exclusion (from 04b): "
          f"removed {n_before - len(summary_df)}, "
          f"{len(summary_df)} clean recordings remain.")
else:
    print("Confirmed-bad list not found (run 04b first) — "
          f"proceeding on all {len(summary_df)} recordings.")


# ============================================================
# 6. PROCESS ALL CLEAN RECORDINGS
# ============================================================

print("\nExtracting oscillation episode features...")
print(f"  Bands: {', '.join(f'{b} {lo}-{hi}Hz' for b,(lo,hi) in BANDS.items())}")
print(f"  Threshold: median + {N_MAD} * MAD")
print(f"  Min episode duration: {MIN_EPISODE_DURATION_SEC} s")
print(f"  Gap-merging tolerance: {MAX_GAP_SEC} s\n")

result_rows = []

for _, row in summary_df.iterrows():
    mouse = int(row["mouse"])
    group = row["group"]
    diff_file = row["differential_file"]
    diff_path = row[path_column]

    print(f"Processing: {diff_file}")

    if not isinstance(diff_path, str) or not os.path.exists(diff_path):
        print("  File not found, skipping.")
        continue

    try:
        signal = np.load(diff_path).astype(np.float64)
    except Exception as e:
        print(f"  Failed to load: {e}")
        continue

    result = {
        "mouse": mouse,
        "group": group,
        "differential_file": diff_file,
    }
    for band_name, (lowcut, highcut) in BANDS.items():
        result.update(
            compute_episode_features(signal, FS, band_name, lowcut, highcut)
        )

    result_rows.append(result)
    print(f"  done | beta:{result.get('beta_n_episodes',0)} "
          f"low_gamma:{result.get('low_gamma_n_episodes',0)} "
          f"high_gamma:{result.get('high_gamma_n_episodes',0)}")

episodes_df = pd.DataFrame(result_rows)


# ============================================================
# 7. SAVE EPISODE FEATURES
# ============================================================

episodes_df.to_csv(EPISODE_FEATURES_PATH, index=False)
print(f"\nSaved episode features:\n{EPISODE_FEATURES_PATH}")


# ============================================================
# 8. MERGE WITH 05B BAND RATIOS (power + ratios + episodes)
# ------------------------------------------------------------
# 05b already holds the clean recording set with metadata, band
# powers and ratios. We left-merge episode features onto it by
# differential_file (unique per recording). mouse/group are dropped
# from the episode table first to avoid duplicate columns.
# ============================================================

if os.path.exists(BAND_RATIOS_PATH):
    ratios = pd.read_csv(BAND_RATIOS_PATH)

    ep_for_merge = episodes_df.drop(columns=["mouse", "group"],
                                    errors="ignore")
    merged = ratios.merge(
        ep_for_merge,
        on="differential_file",
        how="left",
        validate="one_to_one",
    )

    merged.to_csv(FINAL_COMBINED_PATH, index=False)
    print(f"\nSaved FINAL combined table (ratios + episodes):")
    print(FINAL_COMBINED_PATH)
    print(f"Shape: {merged.shape}")

    n_missing = merged[[c for c in merged.columns
                        if c.endswith("_episode_rate")]].isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"Note: {n_missing} recording(s) in 05b had no matching "
              f"episode features (check 03a vs 04a coverage).")
else:
    print(f"\n05b band-ratio file not found at:\n{BAND_RATIOS_PATH}")
    print("Skipping merge step. Run 05b first if you want the combined table.")


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nTotal recordings processed: {len(episodes_df)}")

if len(episodes_df) > 0:
    for band in BANDS:
        cols = [f"{band}_episode_rate", f"{band}_mean_duration_sec",
                f"{band}_mean_amplitude", f"{band}_fraction_of_time"]
        print(f"\nGroup mean {band.upper()} features:")
        print(episodes_df.groupby("group")[cols].mean().round(4))

    print("\nMean episodes detected per recording (median in brackets):")
    for band in BANDS:
        col = f"{band}_n_episodes"
        print(f"  {band:10s}: {episodes_df[col].mean():.1f} "
              f"({episodes_df[col].median():.0f})")

print("\nSTEP 05C finished successfully.")