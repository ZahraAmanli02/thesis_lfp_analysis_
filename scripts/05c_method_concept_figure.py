# ============================================================
# 05C_METHOD_CONCEPT_FIGURE.PY
# Purpose:
# A purely illustrative figure explaining HOW oscillation-episode
# detection works (method, not results). It uses a synthetic signal,
# so it is independent of the data and unaffected by artifacts /
# Mouse 23 — safe to show at any time.
#
# It draws, for one example band:
#   - the band-pass filtered signal (grey)
#   - its Hilbert amplitude envelope (green)
#   - the robust threshold = median + 3*MAD (red dashed)
#   - the detected episodes (shaded), i.e. where the envelope stays
#     above threshold long enough
#
# Output:
#   outputs/05c_method_concept/05c_method_concept.png
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import butter, filtfilt, hilbert


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

OUT_DIR = os.path.join(OUTPUT_DIR, "05c_method_concept")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "05c_method_concept.png")

FS = 1250                 # same sampling rate as the real data
DURATION_SEC = 6          # short window, just for illustration
BAND = (15, 30)           # example band (beta) for the cartoon
N_MAD = 3.0
MIN_DURATION_SEC = 0.1
MAX_GAP_SEC = 0.05

C_SIGNAL = "#888780"      # grey  – filtered signal
C_ENV = "#1D9E75"         # green – envelope
C_THR = "#C44E52"         # red   – threshold
C_EP = "#1D9E75"          # green shading – episodes

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. BUILD A SYNTHETIC SIGNAL WITH A FEW BURSTS
# ============================================================

rng = np.random.default_rng(1)
t = np.arange(0, DURATION_SEC, 1 / FS)
raw = 0.15 * rng.standard_normal(len(t))

# Insert three beta bursts of differing length/strength
bursts = [(1.0, 0.35, 3.0), (2.8, 0.6, 4.0), (4.6, 0.25, 2.5)]
for centre, half_width, amp in bursts:
    m = (t > centre - half_width) & (t < centre + half_width)
    raw[m] += amp * np.sin(2 * np.pi * 22 * t[m])


# ============================================================
# 3. FILTER -> ENVELOPE -> THRESHOLD -> EPISODES
# ============================================================

def bandpass(sig, fs, lo, hi, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


filtered = bandpass(raw, FS, BAND[0], BAND[1])
envelope = np.abs(hilbert(filtered))

med = np.median(envelope)
mad = np.median(np.abs(envelope - med))
threshold = med + N_MAD * mad

above = envelope > threshold
trans = np.diff(above.astype(int))
starts = np.where(trans == 1)[0] + 1
ends = np.where(trans == -1)[0] + 1
if above[0]:
    starts = np.insert(starts, 0, 0)
if above[-1]:
    ends = np.append(ends, len(above))
n = min(len(starts), len(ends))
episodes = list(zip(starts[:n], ends[:n]))

# merge close, drop short
gap = int(MAX_GAP_SEC * FS)
merged = [episodes[0]] if episodes else []
for s, e in episodes[1:]:
    ps, pe = merged[-1]
    if s - pe <= gap:
        merged[-1] = (ps, e)
    else:
        merged.append((s, e))
min_s = int(MIN_DURATION_SEC * FS)
episodes = [(s, e) for s, e in merged if (e - s) >= min_s]


# ============================================================
# 4. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(11, 4.2))

ax.plot(t, filtered, color=C_SIGNAL, lw=0.8, alpha=0.7,
        label="Band-pass filtered signal (beta 15-30 Hz)")
ax.plot(t, envelope, color=C_ENV, lw=1.8, label="Hilbert envelope")
ax.axhline(threshold, color=C_THR, lw=1.4, ls="--",
           label="Threshold = median + 3\u00b7MAD")

for k, (s, e) in enumerate(episodes):
    ax.axvspan(t[s], t[min(e, len(t) - 1)], color=C_EP, alpha=0.12)
    ax.text((t[s] + t[min(e, len(t) - 1)]) / 2, ax.get_ylim()[1] * 0.92,
            f"episode {k + 1}", ha="center", va="top", fontsize=9,
            color="#0F6E56")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Oscillation-episode detection (method illustration)",
             fontsize=12, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower right")

caption = ("An episode is a contiguous stretch where the envelope stays "
           "above threshold for \u2265 0.1 s; near-touching stretches "
           "(gap < 0.05 s) are merged. Synthetic signal \u2014 illustration "
           "of the method only, not real data.")
fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=8,
         color="#444444", wrap=True)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved method concept figure: {OUT_PATH}")
print(f"  Episodes drawn: {len(episodes)}")