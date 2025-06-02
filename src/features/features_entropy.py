import mne
import numpy as np
import pandas as pd
from pathlib import Path
from antropy import sample_entropy
from scipy.signal import welch

# === Define paths ===
CLEAN_DIR = Path("data/clean")
OUTPUT_CSV = Path("outputs/features_entropy.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# === Hjorth parameter functions ===
def hjorth_mobility(signal):
    """
    Hjorth Mobility

    Measures the average frequency of the signal.

    Formula:
        mobility = sqrt(variance of the first derivative / variance of the signal)

    ---

    Step-by-step:
    1. np.diff(signal) computes the difference between each point and the next:
       For example, if signal = [1, 4, 9], then np.diff(signal) = [3, 5]
       This gives us an approximation of the signal's first derivative (i.e., how fast it's changing).

    2. np.var(x) calculates the variance of x:
       This measures how spread out the values are from their average.
       Higher variance = more fluctuation.

    3. So:
       - np.var(deriv) tells us how wildly the signal changes
       - np.var(signal) tells us how "big" or "intense" the signal is overall

    Interpretation:
        - Low mobility: slow, smooth changes (e.g., rest state)
        - High mobility: fast, frequent changes (e.g., motor activity)
    """
    deriv = np.diff(signal)
    return np.sqrt(np.var(deriv) / np.var(signal))


def hjorth_complexity(signal):
    """
    Hjorth Complexity

    Measures how rapidly the signal’s frequency changes — i.e., how irregular or chaotic it is.

    Formula:
        complexity = sqrt(variance of second derivative / variance of first derivative) / mobility

    ---

    Step-by-step:
    1. np.diff(signal): approximates the first derivative (rate of change of signal)
    2. np.diff(np.diff(signal)): approximates the second derivative (rate of change *of* rate of change)

       Example:
         signal = [1, 4, 9] → np.diff(signal) = [3, 5] → np.diff([3, 5]) = [2]
         This tells us how sharply the changes themselves are changing.

    3. np.var(x): measures the variance (how much the values fluctuate)

    4. Then we divide the variance of the 2nd derivative by the 1st,
       and normalize it by the mobility (to make it scale-invariant)

    Interpretation:
        - Low complexity = smooth frequency pattern (e.g., constant rhythm)
        - High complexity = rapidly changing frequency → indicates more cognitive load or movement
    """
    deriv1 = np.diff(signal)
    deriv2 = np.diff(deriv1)
    return np.sqrt(np.var(deriv2) / np.var(deriv1)) / hjorth_mobility(signal)

# === Extract features from each file ===
features = []
fif_files = sorted(CLEAN_DIR.glob("*.fif"))

for fif_file in fif_files:
    print(f"Processing {fif_file.name}")
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
    data, _ = raw[:, :]  # Get EEG signal, shape = (n_channels, n_times)

    # === Reduce to 1D signal by averaging over all EEG channels ===
    # This gives a single general-purpose signal representing overall brain activity over time
    signal = np.mean(data, axis=0)

    # === Sample Entropy ===
    #
    # What is sample_entropy?
    # Sample Entropy measures how self-similar the signal is — i.e., how often patterns repeat.
    #
    # - If the signal contains many repeated patterns → entropy is low (more predictable)
    # - If the signal is highly irregular and diverse → entropy is high (less predictable)
    #
    # 🔬 How does it work?
    # sample_entropy(signal):
    # 1. It slides a window over the signal (e.g., of size 2 or 3 values)
    # 2. It checks how often those small patterns occur throughout the signal
    # 3. Then it compares whether those patterns also match in the next time step
    # 4. If many matches are found → low entropy (signal is repetitive)
    # 5. If few matches are found → high entropy (signal is complex)
    #
    # This feature is useful for EEG because:
    # - Resting brain activity tends to be more repetitive → lower entropy
    # - Active or motor/cognitive tasks often produce more complex signals → higher entropy
    try:
        sampen = sample_entropy(signal)
    except:
        sampen = 0  # fallback if NaN or error


    # === Hjorth Mobility ===
    # Measures the average frequency of the signal.
    # Computed as: sqrt(variance of first derivative / variance of signal)
    # Higher values → faster fluctuations
    mob = hjorth_mobility(signal)

    # === Hjorth Complexity ===
    # Measures how rapidly the frequency content of the signal changes.
    # Computed as: sqrt(var(second derivative) / var(first derivative)) / mobility
    # Higher values → more irregular frequency shifts
    comp = hjorth_complexity(signal)

    # === Frequency bands using Welch PSD ===
    freqs, psd = welch(signal, fs=raw.info["sfreq"], nperseg=1024)
    alpha_idx = np.where((freqs >= 8) & (freqs < 13))[0]
    beta_idx = np.where((freqs >= 13) & (freqs < 30))[0]

    alpha_power = np.mean(psd[alpha_idx])
    beta_power = np.mean(psd[beta_idx])
    alpha2_beta = (alpha_power**2) / beta_power if beta_power > 0 else 0

    # === Metadata ===
    parts = fif_file.stem.split("_")
    subject = parts[0]
    run = parts[1] if len(parts) > 1 else "unknown"

    # === Label assignment ===
    if "R01" in run or "R02" in run:
        label = "rest"
    elif any(r in run for r in ["R03", "R04", "R07", "R08"]):
        label = "motor"
    else:
        label = "unknown"

    # === Save extracted features ===
    features.append({
        "subject": subject,
        "run": run,
        "label": label,
        "sample_entropy": sampen,
        "hjorth_mobility": mob,
        "hjorth_complexity": comp,
        "alpha2_over_beta": alpha2_beta
    })

# === Save to CSV ===
df = pd.DataFrame(features)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Features saved to {OUTPUT_CSV}")
