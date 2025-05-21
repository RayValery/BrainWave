import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Load a cleaned EEG file ===
file_path = Path("data/clean/S001_S001R01.fif")
raw = mne.io.read_raw_fif(file_path, preload=True)

# === Compute PSD (1–30 Hz) using Welch method ===
psd = raw.compute_psd(fmin=1, fmax=30)
freqs = psd.freqs
psd_values = psd.get_data()  # <- обов’язково викликати .get_data()

# === Average across channels ===
mean_psd = np.mean(psd_values, axis=0)

# === Define EEG frequency bands ===
bands = {
    "Delta (1–4 Hz)": (1, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–13 Hz)": (8, 13),
    "Beta (13–30 Hz)": (13, 30)
}

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(freqs, mean_psd, label="Mean PSD", color='black')

colors = ['#d0f0c0', '#add8e6', '#fceabb', '#f4cccc']
for (label, (fmin, fmax)), color in zip(bands.items(), colors):
    plt.axvspan(fmin, fmax, color=color, alpha=0.4, label=label)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (a.u.)")
plt.title(f"PSD of {file_path.name}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
