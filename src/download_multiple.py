import mne
from pathlib import Path
import shutil

# 🔧 Parameters — adjust these as needed
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]       # participants S001, S002, S003 - S010
runs = [1, 2, 3, 7, 8]                           # Run 1,2 = rest (eyes open), Run 3,7,8 = motor imagery

# 💾 Target directory to organize data in your project
target_dir = Path("data/raw")
target_dir.mkdir(parents=True, exist_ok=True)

print(f"📥 Downloading EEG data for subjects {subjects} and runs {runs}...\n")

for subject in subjects:
    print(f"🔹 Subject {subject:03d}")
    edf_paths = mne.datasets.eegbci.load_data(subject=subject, runs=runs)

    for edf_path in edf_paths:
        edf_path = Path(edf_path)
        new_filename = f"{edf_path.parent.name}_{edf_path.name}"  # e.g. S001_S001R01.edf
        dest_path = target_dir / new_filename

        # ✅ Copy to local project folder
        shutil.copy(edf_path, dest_path)
        print(f"   ↪ Copied {edf_path.name} → {dest_path.name}")

print("\n✅ All files downloaded and copied to data/raw/")
