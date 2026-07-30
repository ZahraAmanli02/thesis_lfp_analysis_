# ============================================================
# MAKE_DEFENSE_LFP_SNIPPET.PY
#
# Generate a clean, defense-ready LFP trace snippet for Slide 2.
# Reads a differential .npy from 03a and saves a 5-second window
# with large fonts, no borders, transparent background — ready to
# drop into PowerPoint.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- SETTINGS ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPY_PATH = os.path.join(
    BASE_DIR, "outputs", "03a_differential_npy_filtered_Cable1",
    "Mouse14_Cable1_HF_2026-03-29_13-48-59_DIFF_Ch2_minus_Ch3.npy"
    
)
FS = 1250.0                     # sampling rate in Hz (matches 04a)
WINDOW_START_SEC = 6.0          # start of the snippet — try 10, 15, 20 for cleaner windows
WINDOW_LEN_SEC = 4.0            # 4 s trims the tail-end artifact seen in the first pass

OUT_DIR = os.path.join(BASE_DIR, "outputs", "defense_slide_visuals")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "slide2_lfp_snippet.png")


# ---------- LOAD + WINDOW ----------
signal = np.load(NPY_PATH)
if signal.ndim > 1:
    signal = signal.ravel()

i0 = int(WINDOW_START_SEC * FS)
i1 = int((WINDOW_START_SEC + WINDOW_LEN_SEC) * FS)
snippet = signal[i0:i1]
t = np.arange(len(snippet)) / FS


# ---------- PLOT ----------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.5,
})

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(t, snippet, color="#1f4e79", linewidth=1.0)
ax.set_xlabel("Time (s)", fontsize=18)
ax.set_ylabel("Amplitude (µV)", fontsize=18)
ax.tick_params(axis="both", labelsize=15, width=1.5, length=6)
ax.set_xlim(0, WINDOW_LEN_SEC)
ax.grid(False)

# Optional: subtle title
ax.set_title("Example LFP  |  Lateral hypothalamus  |  Mouse 14 (HF)",
             fontsize=17, pad=12, loc="left", color="#333333")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight",
            facecolor="white", transparent=False)
plt.close(fig)
print(f"Saved defense-ready LFP snippet:\n{OUT_PNG}")
print(f"Window: {WINDOW_START_SEC:.1f} – {WINDOW_START_SEC + WINDOW_LEN_SEC:.1f} s")
print(f"Fs used: {FS} Hz  (change FS in script if this is wrong)")
