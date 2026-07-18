import matplotlib.pyplot as plt


def plot_fit(xs, sig_detrended, fit):
    """
    Plots the Gaussian fit over the detrended ECG signal.

    Parameters:
    - xs: array-like, x-axis values (index)
    - sig_detrended: array-like, the detrended ECG signal

    """

    plt.figure(figsize=(8, 5))  
    plt.plot(xs, sig_detrended, color="dodgerblue", linewidth=2, label='Detrended Signal')
    plt.plot(xs, fit, color="mediumvioletred", linestyle='--', linewidth=2, label='Gaussian Fit')

    plt.title('Gaussian Fitting to ECG Signal', fontsize=14, weight='bold')
    plt.xlabel('Cycle-Relative Index', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()
