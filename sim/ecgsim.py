import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

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
    plot=True
):
    """
    Generate an ECG signal with optional noise, drift, and line noise.

    Parameters:
        duration (int): Duration of the signal in seconds.
        sampling_rate (int): Sampling rate in Hz.
        heart_rate (int): Heart rate in beats per minute.
        noise_level (float): Standard deviation of the noise.
        drift_start (float): Starting value of the drift.
        drift_end (float): Ending value of the drift.
        line_noise_frequency (float): Frequency of the line noise in Hz.
        line_noise_amplitude (float): Amplitude of the line noise.
        start_time (float): Start time of the plot interval in seconds.
        end_time (float): End time of the plot interval in seconds.
        random_seed (int): Seed for random number generator.
        plot (bool): Whether to plot the signal.

    Returns:
        tuple: Generated ECG signal with noise and drift, time array, and plot interval indices.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Simulate ECG signal
    ecg_signal = nk.ecg_simulate(
        duration=duration, 
        sampling_rate=sampling_rate, 
        heart_rate=heart_rate, 
        random_state=random_seed
    )

    # Add noise, drift, and line noise
    noise = np.random.normal(0, noise_level * np.std(ecg_signal), len(ecg_signal))
    drift = np.linspace(drift_start, drift_end, len(ecg_signal))
    time = np.arange(len(ecg_signal)) / sampling_rate
    line_noise = line_noise_amplitude * np.sin(2 * np.pi * line_noise_frequency * time)

    noisy_ecg_with_drift = ecg_signal + noise + drift + line_noise

    # Determine plot interval
    start_time = int(start_time * sampling_rate)
    end_time = int(end_time * sampling_rate)
    time_axis = time[start_time:end_time]

    # Plot if requested
    if plot:
        # Full signal plot
        plt.figure(figsize=(12, 3))
        plt.plot(time, noisy_ecg_with_drift, label="Generated ECG Signal", color="dodgerblue", linewidth=1.5, alpha=0.8)
        plt.title("Full ECG Signal with Noise, Drift, and Line Noise", fontsize=14, fontweight="bold")
        plt.xlabel("Time (s)", fontsize=12, fontweight="bold")
        plt.ylabel("Amplitude", fontsize=12, fontweight="bold")
        plt.legend(loc="upper right", fontsize=11)
        plt.tight_layout()
        plt.show()

        # Interval plot
        plt.figure(figsize=(12, 3))
        plt.plot(time_axis, noisy_ecg_with_drift[start_time:end_time], label="Generated ECG Signal", color="dodgerblue", linewidth=1.5, alpha=0.8)
        plt.title("ECG Signal with Noise, Drift, and Line Noise (Interval)", fontsize=14, fontweight="bold")
        plt.xlabel("Time (s)", fontsize=12, fontweight="bold")
        plt.ylabel("Amplitude", fontsize=12, fontweight="bold")
        plt.legend(loc="upper right", fontsize=11)
        plt.tight_layout()
        plt.show()

    return noisy_ecg_with_drift, sampling_rate, time, time_axis, start_time, end_time
