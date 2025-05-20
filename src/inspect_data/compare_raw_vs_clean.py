import mne
from pathlib import Path
import matplotlib.pyplot as plt

# === File paths ===
edf_file = Path("data/raw/S001_S001R01.edf")            # raw .edf
clean_file = Path("data/clean/S001_S001R01.fif")    # cleaned .fif

# === Load raw (unfiltered) ===
raw_raw = mne.io.read_raw_edf(edf_file, preload=True)
raw_raw.set_eeg_reference('average', projection=True)

# === Load cleaned ===
raw_clean = mne.io.read_raw_fif(clean_file, preload=True)

# === Plot time series comparison ===
print("🔍 Plotting raw EEG (unfiltered)...")
raw_raw.plot(n_channels=10, duration=10, title="Raw EEG (unfiltered)")

print("🔍 Plotting cleaned EEG...")
raw_clean.plot(n_channels=10, duration=10, title="Cleaned EEG")

# === Plot PSD comparison side-by-side ===
psd_raw = raw_raw.compute_psd(fmax=60)
psd_clean = raw_clean.compute_psd(fmax=60)

fig_raw = psd_raw.plot(show=False)
fig_clean = psd_clean.plot(show=False)

# Save both to files
fig_raw.savefig("outputs/psd_raw_S001.png")
fig_clean.savefig("outputs/psd_clean_S001.png")

print("✅ Saved PSD plots to outputs/")
