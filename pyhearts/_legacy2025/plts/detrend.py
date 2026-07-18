import matplotlib.pyplot as plt


def plot_detrended_cycle(xs, sig, sig_corrected, cycle):
    """
    Plot the original and baseline-corrected ECG signals.

    Parameters:
    - xs (np.array): Array of sample indices corresponding to the ECG signal.
    - sig (np.array): Original detrended ECG signal.
    - sig_corrected (np.array): Baseline-corrected ECG signal.
    - cycle (int): Cycle number for labeling the plot.

    Returns:
    - None: Displays the plot for the given ECG cycle.
    """
    cycle_start = xs[0]  
    xs_relative = xs - cycle_start 

    plt.figure(figsize=(8, 5))
    
    # Plot original signal (detrended)
    plt.plot(xs_relative, sig, label='Original Signal (Detrended)', color='gray', linestyle='--')
    
    # Plot baseline-corrected signal
    plt.plot(xs_relative, sig_corrected, label='Baseline Corrected Signal', color='dodgerblue')

    # Labels and title
    plt.xlabel("Cycle-Relative Index", fontsize=12)
    plt.ylabel("ECG Signal (mV)", fontsize=12)
    plt.title(f"Original vs. Baseline Corrected Signal - Cycle {cycle}", fontsize=14, weight='bold')
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.show()

