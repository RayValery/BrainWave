import mne
import numpy as np
import pandas as pd
from pathlib import Path

# === Define EEG frequency bands ===
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}

# === Set input and output paths ===
CLEAN_DIR = Path("data/clean")                        # Folder with preprocessed .fif files
OUTPUT_CSV = Path("outputs/features_advanced.csv")    # Output feature file
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# === Get list of EEG files ===
fif_files = sorted(CLEAN_DIR.glob("*.fif"))
features = []

# === Process each EEG file ===
for fif_file in fif_files:
    print(f"Processing {fif_file.name}")
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
    psd = raw.compute_psd(fmin=1, fmax=30)
    freqs = psd.freqs
    psd_values = psd.get_data()  # shape: (n_channels, n_freqs)
    mean_psd = np.mean(psd_values, axis=0)

    # === Extract mean power for each frequency band ===
    band_powers = {}
    for band, (fmin, fmax) in FREQ_BANDS.items():
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        band_powers[band] = np.mean(mean_psd[idx])

    # === Compute derived features ===
    alpha = band_powers["alpha"]
    theta = band_powers["theta"]
    beta = band_powers["beta"]
    delta = band_powers["delta"]
    total = sum(band_powers.values())

    band_powers.update({
        "alpha_theta_ratio": alpha / theta if theta > 0 else 0,
        "beta_alpha_ratio": beta / alpha if alpha > 0 else 0,
        "total_power": total,
        "log_alpha": np.log(alpha) if alpha > 0 else 0,
        "spectral_ratio": (alpha + beta) / (theta + delta) if (theta + delta) > 0 else 0
    })

    # === Parse subject ID and run name from file name ===
    parts = fif_file.stem.split("_")
    subject = parts[0]
    run = parts[1] if len(parts) > 1 else "unknown"

    # === Assign label based on run name ===
    if "R01" in run or "R02" in run:
        label = "rest"
    elif any(r in run for r in ["R03", "R04", "R07", "R08"]):
        label = "motor"
    else:
        label = "unknown"

    # === Store all features for this file ===
    features.append({
        "subject": subject,
        "run": run,
        "label": label,
        **band_powers
    })

# === Convert to DataFrame and save ===
df = pd.DataFrame(features)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Features saved to: {OUTPUT_CSV}")
