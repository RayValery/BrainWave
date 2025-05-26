import mne
import numpy as np
import pandas as pd
from pathlib import Path

# EEG frequency bands
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}

# Paths
CLEAN_DIR = Path("data/clean")
OUTPUT_CSV = Path("outputs/features.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Collect all .fif files
fif_files = sorted(CLEAN_DIR.glob("*.fif"))

features = []

for fif_file in fif_files:
    print(f"🔍 Extracting features from {fif_file.name}")
    raw = mne.io.read_raw_fif(fif_file, preload=True)

    # Compute power spectral density
    psd = raw.compute_psd(fmin=1, fmax=30)
    freqs = psd.freqs
    psd_values = psd.get_data()  # shape: (n_channels, n_freqs)

    # Average across all EEG channels
    mean_psd = np.mean(psd_values, axis=0)  # shape: (n_freqs,)

    # Compute average power in each frequency band
    band_powers = {}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        idx_band = np.where((freqs >= fmin) & (freqs < fmax))[0]  # знайди всі частоти в межах (fmin, fmax) і поверни їхні індекси, наприклад: [6, 7, 8, 9, 10, 11, 12, 13] (це індекси частот 4.0, 4.5, ..., 7.5)
                                                                  # (freqs >= fmin) & (freqs < fmax) --> створює булевий масив: True там, де частота входить в діапазон
                                                                  # np.where(...) --> повертає індекси, де True
                                                                  # [0]	--> витягує самі індекси (без обгортки у tuple)
        band_power = np.mean(mean_psd[idx_band])
        band_powers[band_name] = band_power

    # Metadata from filename
    parts = fif_file.stem.split("_")  # e.g. S001_S001R01
    subject = parts[0]
    run = parts[1] if len(parts) > 1 else "unknown"

    # Assign label based on run number
    if "R01" in run or "R02" in run:
        label = "rest"
    elif any(r in run for r in ["R03", "R04", "R07", "R08"]):
        label = "motor"
    else:
        label = "unknown"

    # Append result
    features.append({
        "subject": subject,
        "run": run,
        "label": label,
        **band_powers
    })

# Convert to DataFrame and save
df = pd.DataFrame(features)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Features saved to {OUTPUT_CSV}")
