from typing import Optional, Union, Sequence
import numpy as np
import matplotlib.pyplot as plt


def plot_rpeaks(
    ecg_signal: Union[np.ndarray, Sequence[float]],
    sampling_rate: float,
    r_peaks: Union[np.ndarray, Sequence[int]],
    *,
    crop_ms: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "ECG Signal with Annotated R-Peaks",
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude",
    line_width: float = 1.5,
    show: bool = True,
) -> plt.Axes:
    """
    Plot the ECG signal with annotated R-peaks, optionally cropped to an initial time window.

    Parameters
    ----------
    ecg_signal : array-like
        1D ECG signal.
    sampling_rate : float
        Sampling rate in Hz (> 0).
    r_peaks : array-like
        Sample indices of detected R-peaks.
    crop_ms : int, optional
        If provided, plot only the first `crop_ms` milliseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure/axes is created.
    title : str, default "ECG Signal with Annotated R-Peaks"
        Plot title.
    xlabel : str, default "Time (s)"
        X label.
    ylabel : str, default "Amplitude"
        Y label.
    line_width : float, default 1.5
        Line width for the ECG trace.
    show : bool, default True
        If True, calls `plt.show()` at the end.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.
    """
    # --- Coerce & validate ---
    sig = np.asarray(ecg_signal, dtype=float)
    if sig.ndim != 1 or sig.size == 0:
        raise ValueError("`ecg_signal` must be a non-empty 1D array.")
    if sampling_rate <= 0:
        raise ValueError("`sampling_rate` must be > 0.")

    r_peaks = np.asarray(r_peaks, dtype=int)
    if r_peaks.ndim != 1:
        raise ValueError("`r_peaks` must be a 1D array of indices.")

    # --- Time axis ---
    t = np.arange(sig.size) / sampling_rate

    # --- Optional crop ---
    if crop_ms is not None:
        if crop_ms <= 0:
            raise ValueError("`crop_ms` must be positive when provided.")
        crop_samples = int(round(crop_ms * sampling_rate / 1000.0))
        crop_samples = max(1, min(crop_samples, sig.size))
        sig = sig[:crop_samples]
        t = t[:crop_samples]
        r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < crop_samples)]

    # Guard: filter out-of-bounds R-peaks (in case inputs are messy)
    r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < sig.size)]

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(t, sig, label="ECG Signal", linewidth=line_width, alpha=0.9)

    if r_peaks.size > 0:
        ax.scatter(t[r_peaks], sig[r_peaks], label="R-peaks", zorder=3, color = 'red')

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Legend (only if we have at least one handle)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Deduplicate while preserving order
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        ax.legend(*zip(*uniq), loc="upper right", fontsize=11)

    plt.tight_layout()
    if show:
        plt.show()

    return ax
