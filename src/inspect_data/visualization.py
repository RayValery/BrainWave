from pathlib import Path
import mne
import matplotlib.pyplot as plt


def plot_psd(fif_file: Path, fmax: float = 60.0, save_path: Path = None):
    """
    Compute and plot the Power Spectral Density (PSD) of a cleaned EEG file.

    :param fif_file: Path to a .fif file (cleaned EEG)
    :param fmax: Max frequency to show in the plot
    :param save_path: Optional path to save the plot as PNG
    """
    raw = mne.io.read_raw_fif(fif_file, preload=True)
    psd = raw.compute_psd(fmax=fmax)
    fig = psd.plot(show=False)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"✅ PSD plot saved to {save_path}")
    else:
        plt.show()


def compare_psd(edf_file: Path, fif_file: Path, fmax: float = 60.0, save_prefix: Path = None):
    """
    Compare PSDs of raw (.edf) and cleaned (.fif) EEG data side-by-side.

    :param edf_file: Path to the original raw .edf EEG file
    :param fif_file: Path to the cleaned .fif EEG file
    :param fmax: Max frequency to show in the plot
    :param save_prefix: Optional path prefix to save both plots
    """
    raw_raw = mne.io.read_raw_edf(edf_file, preload=True)
    raw_raw.set_eeg_reference('average', projection=True)

    raw_clean = mne.io.read_raw_fif(fif_file, preload=True)

    # Compute PSDs
    psd_raw = raw_raw.compute_psd(fmax=fmax)
    psd_clean = raw_clean.compute_psd(fmax=fmax)

    # Plot
    fig_raw = psd_raw.plot(show=False)
    fig_clean = psd_clean.plot(show=False)

    if save_prefix:
        Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)
        fig_raw.savefig(f"{save_prefix}_raw.png")
        fig_clean.savefig(f"{save_prefix}_clean.png")
        print(f"✅ Saved comparison plots: {save_prefix}_raw.png, {save_prefix}_clean.png")
    else:
        plt.show()
