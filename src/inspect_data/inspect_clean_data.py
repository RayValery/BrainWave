import mne
from pathlib import Path

# === Load a cleaned EEG file ===
clean_file = Path("data/clean/S001_S001R01.fif")  # Replace with your file name

print(f"Loading {clean_file.name}...")
raw = mne.io.read_raw_fif(clean_file, preload=True)

# === Plot EEG time series ===
print("Showing EEG signal (first 10 channels)...")
raw.plot(n_channels=10, duration=10, scalings='auto')  # Interactive window

# === Plot Power Spectral Density ===
print("Showing Power Spectral Density (0–60 Hz)...")
psd = raw.compute_psd(fmax=60)

# Plot and save as PNG
fig = psd.plot()
fig.savefig("outputs/psd_S001_S001R01.png")
print("✅ PSD plot saved to outputs/psd_S001_S001R01.png")
