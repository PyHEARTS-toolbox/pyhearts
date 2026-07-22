from typing import Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

def plot_labeled_peaks(
    xs: Union[np.ndarray, list[float]],
    signal: Union[np.ndarray, list[float]],
    peak_data: Dict[str, Dict[str, Optional[int]]],
    *,
    ax: Optional[plt.Axes] = None,
    peak_colors: Optional[Dict[str, str]] = None,
    title: str = "Detected ECG Peaks",
    xlabel: str = "Cycle-Relative Index",
    ylabel: str = "Amplitude",
    label_offset: float = 0.05,
    pad_y: float = 0.1,
    show: bool = True,
) -> plt.Axes:
    """
    Plot an ECG signal with labeled P, Q, R, S, and T peaks.

    Parameters
    ----------
    xs : array-like
        1D x-axis values for the signal (time or sample index).
    signal : array-like
        1D detrended ECG signal values.
    peak_data : dict[str, dict[str, Optional[int]]]
        Mapping from component label (e.g., "P") to a dict with key
        "center_idx" indicating the peak's sample index in `signal`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates a new figure and axes.
    peak_colors : dict[str, str], optional
        Colors per label (e.g., {"P":"orange", "Q":"green"}). Defaults are used if None.
    title : str, default "Detected ECG Peaks"
        Plot title.
    xlabel : str, default "Cycle-Relative Index"
        X-axis label.
    ylabel : str, default "Amplitude"
        Y-axis label.
    label_offset : float, default 0.05
        Vertical offset (in signal units) for peak text labels.
    pad_y : float, default 0.1
        Fractional padding to add to y-limits.
    show : bool, default True
        If True, calls `plt.show()` at the end.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot drawn.
    """
    xs = np.asarray(xs)
    signal = np.asarray(signal, dtype=float)

    if xs.ndim != 1 or signal.ndim != 1 or xs.size != signal.size:
        raise ValueError("`xs` and `signal` must be 1D arrays of equal length.")

    # Default peak colors
    if peak_colors is None:
        peak_colors = {"P": "orange", "Q": "green", "R": "red", "S": "purple", "T": "magenta"}

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the main ECG signal
    ax.plot(xs, signal, color="dodgerblue", linewidth=2, label="Detrended Signal", zorder=1)

    # Plot peaks
    for label, data in peak_data.items():
        if "center_idx" not in data:
            continue
        idx = data["center_idx"]
        if idx is None or not (0 <= idx < signal.size):
            continue
        color = peak_colors.get(label, "black")

        ax.scatter(xs[idx], signal[idx], color=color, label=f"{label} Peak", zorder=3, edgecolor="white")
        ax.text(
            xs[idx],
            signal[idx] + label_offset,
            label,
            color=color,
            fontsize=12,
            ha="center",
            fontweight="bold",
            zorder=4,
        )

    # Adjust y-axis limits
    y_min, y_max = np.nanmin(signal), np.nanmax(signal)
    rng = max(y_max - y_min, np.finfo(float).eps)
    ax.set_ylim(y_min - pad_y * rng, y_max + pad_y * rng)

    # Labels & legend
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)
    if uniq_handles:
        ax.legend(uniq_handles, uniq_labels, fontsize=11, loc="best")

    plt.tight_layout()
    if show:
        plt.show()

    return ax
