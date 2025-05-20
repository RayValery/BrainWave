import mne
from mne.io import read_raw_edf
import matplotlib.pyplot as plt

# 1. Select subject and run to download
subject = 1
run = 1

print(f"Downloading EEG data for subject {subject}, run {run}...")

# 2. Download EEG data from PhysioNet (Motor Movement/Imagery dataset)
data_paths = mne.datasets.eegbci.load_data(
    subject=subject,
    runs=[run],
    path="../data",           # Save files to the data/ folder
    update_path=True
)

print(f"Download completed. File path(s): {data_paths}")

# 3. Read the first EDF file (you can expand this to load all runs if needed)
edf_file = data_paths[0]
raw = read_raw_edf(edf_file, preload=True)

# 4. Set EEG reference to average — improves signal clarity for some tasks
raw.set_eeg_reference('average', projection=True)

# 5. Show basic recording info
print("\nEEG Info:")
print(raw.info)

# 6. Plot EEG data — first 10 channels, 10 seconds
print("\nOpening EEG plot...")
raw.plot(n_channels=10, duration=10, scalings='auto')
