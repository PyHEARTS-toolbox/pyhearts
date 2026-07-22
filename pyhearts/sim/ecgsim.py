import numpy as np
import neurokit2 as nk


def generate_ecg_signal(
    duration=110,
    sampling_rate=1000,
    heart_rate=60,
    noise_level=0.1,
    drift_start=-0.5,
    drift_end=0.5,
    line_noise_frequency=50,
    line_noise_amplitude=0.05,
    start_time=21.5,
    end_time=25,
    random_seed=2,
    plot=False,
):
    """
    Generate a synthetic ECG with optional noise, drift, and line interference.

    Requires the optional ``sim`` extra (``pip install "pyhearts[sim]"``).
    For a minimal demo signal, NeuroKit2's ``ecg_simulate`` (as used in
    ``examples/demo.ipynb``) is often simpler.

    Parameters
    ----------
    duration : int, default 110
        Signal length in seconds.
    sampling_rate : int, default 1000
        Sampling rate in Hz.
    heart_rate : int, default 60
        Heart rate in beats per minute.
    noise_level : float, default 0.1
        Standard deviation of additive Gaussian noise, scaled by the clean
        signal standard deviation.
    drift_start, drift_end : float
        Endpoints of a linear baseline drift added to the signal.
    line_noise_frequency : float, default 50
        Mains-line noise frequency in Hz.
    line_noise_amplitude : float, default 0.05
        Amplitude of the sinusoidal line noise.
    start_time, end_time : float
        Plot window bounds in seconds (used only when ``plot=True``).
    random_seed : int, default 2
        Seed for NumPy and NeuroKit2 simulation.
    plot : bool, default False
        If True, display full-signal and zoomed matplotlib figures.

    Returns
    -------
    noisy_ecg_with_drift : np.ndarray
        Simulated ECG with noise/drift/line interference.
    sampling_rate : int
        Echo of the requested sampling rate.
    time : np.ndarray
        Full time axis in seconds.
    time_axis : np.ndarray
        Time axis for the optional plot window.
    start_idx, end_idx : int
        Sample indices corresponding to ``start_time`` / ``end_time``.
    """
    np.random.seed(random_seed)

    ecg_signal = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
        random_state=random_seed,
    )

    noise = np.random.normal(0, noise_level * np.std(ecg_signal), len(ecg_signal))
    drift = np.linspace(drift_start, drift_end, len(ecg_signal))
    time = np.arange(len(ecg_signal)) / sampling_rate
    line_noise = line_noise_amplitude * np.sin(2 * np.pi * line_noise_frequency * time)

    noisy_ecg_with_drift = ecg_signal + noise + drift + line_noise

    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)
    time_axis = time[start_idx:end_idx]

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 3))
        plt.plot(
            time,
            noisy_ecg_with_drift,
            label="Generated ECG Signal",
            color="dodgerblue",
            linewidth=1.5,
            alpha=0.8,
        )
        plt.title("Full ECG Signal with Noise, Drift, and Line Noise", fontsize=14, fontweight="bold")
        plt.xlabel("Time (s)", fontsize=12, fontweight="bold")
        plt.ylabel("Amplitude", fontsize=12, fontweight="bold")
        plt.legend(loc="upper right", fontsize=11)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 3))
        plt.plot(
            time_axis,
            noisy_ecg_with_drift[start_idx:end_idx],
            label="Generated ECG Signal",
            color="dodgerblue",
            linewidth=1.5,
            alpha=0.8,
        )
        plt.title(
            "ECG Signal with Noise, Drift, and Line Noise (Interval)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Time (s)", fontsize=12, fontweight="bold")
        plt.ylabel("Amplitude", fontsize=12, fontweight="bold")
        plt.legend(loc="upper right", fontsize=11)
        plt.tight_layout()
        plt.show()

    return noisy_ecg_with_drift, sampling_rate, time, time_axis, start_idx, end_idx
