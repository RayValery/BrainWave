import mne
from pathlib import Path

# 🔧 Parameters
RAW_DATA_DIR = Path("data/raw")
CLEAN_DATA_DIR = Path("data/clean")
CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Bandpass filter settings (1–40 Hz)
LOW_FREQ = 1.
HIGH_FREQ = 40.
NOTCH_FREQ = 50.  # Hz

# Find all .edf files in data/raw
edf_files = list(RAW_DATA_DIR.glob("*.edf"))

print(f"🧠 Found {len(edf_files)} EDF file(s) to preprocess...\n")

for edf_path in edf_files:
    print(f"🔹 Processing {edf_path.name}")

    # Load raw EEG data
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # Set EEG reference to average
    raw.set_eeg_reference('average', projection=True)

    # Apply notch filter at 50 Hz (remove power line noise)
    raw.notch_filter(freqs=NOTCH_FREQ)

    # Apply bandpass filter to keep only 1–40 Hz activity
    raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ)

    # Save the cleaned data in .fif format
    output_path = CLEAN_DATA_DIR / edf_path.with_suffix(".fif").name
    raw.save(output_path, overwrite=True)

    print(f"   ✅ Saved cleaned file to {output_path.name}\n")

print("✅ All files processed and saved in data/clean/")
