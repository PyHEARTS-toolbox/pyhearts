import matplotlib.pyplot as plt
import numpy as np


def plot_dynamic_offset(
    xs: np.ndarray,
    sig: np.ndarray,
    r_center_idx: int,
    r_left_idx: int,
    r_right_idx: int,
    q_min_idx: int,
    s_max_idx: int
):
    """
    Visualize precomputed dynamic offset zones for Q and S peak detection.

    Parameters
    ----------
    xs : np.ndarray
        X-axis values (sample indices).
    sig : np.ndarray
        Detrended ECG signal.
    r_center_idx : int
        Index of the R peak center (for vertical marker).
    r_left_idx : int
        Left bound of the R region (R - k·σ).
    r_right_idx : int
        Right bound of the R region (R + k·σ).
    q_min_idx : int
        Start index for Q peak search.
    s_max_idx : int
        End index for S peak search.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(xs, sig, color="dodgerblue", linewidth=2, label="Detrended Signal")

    # R peak bounds
    plt.axvline(xs[r_center_idx], color="black", linestyle="-", linewidth=1.5, label="R peak")
    plt.axvline(xs[r_left_idx], color="orange", linestyle="--", linewidth=1.5, label="R - k·σ")
    plt.axvline(xs[r_right_idx], color="red", linestyle="--", linewidth=1.5, label="R + k·σ")

    # Q/S dynamic offset bounds
    plt.axvline(xs[q_min_idx], color="green", linestyle="--", linewidth=1.5, label="Q search start")
    plt.axvline(xs[s_max_idx], color="purple", linestyle="--", linewidth=1.5, label="S search end")

    # Shaded regions
    plt.axvspan(xs[q_min_idx], xs[r_left_idx], color="yellow", alpha=0.3, label="Q offset zone")
    plt.axvspan(xs[r_right_idx], xs[s_max_idx], color="yellow", alpha=0.3, label="S offset zone")

    plt.title("Dynamic Offset Zones for Q and S Peak Detection", fontsize=14, weight="bold")
    plt.xlabel("Relative Sample Index", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)

    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10)

    plt.tight_layout()
    plt.show()

