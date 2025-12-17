from typing import Dict, Optional, Mapping
import numpy as np
import matplotlib.pyplot as plt

def plot_fwhm(
    xs: np.ndarray,
    signal: np.ndarray,
    peak_inds: Dict[str, Optional[int]],
    fwhm_results: Dict[str, Mapping[str, Optional[int]]],
    *,
    ax: Optional[plt.Axes] = None,
    peak_colors: Optional[Dict[str, str]] = None,
    title: str = "Detected Peaks and FWHM Boundaries",
    xlabel: str = "Cycle-Relative Index",
    ylabel: str = "Amplitude",
    pad_y: float = 0.1,
    show: bool = True,
) -> plt.Axes:
    """
    Plot ECG segment with detected peaks and FWHM (left/right) boundaries.

    Parameters
    ----------
    xs : np.ndarray
        1D array of x-axis values (time or sample index).
    signal : np.ndarray
        1D array of detrended ECG amplitudes aligned to `xs`.
    peak_inds : dict[str, Optional[int]]
        Mapping from component label (e.g., "P","Q","R","S","T") to center index in `signal`.
    fwhm_results : dict[str, Mapping[str, Optional[int]]]
        For each label, a mapping containing keys like {"fwhm_left": int, "fwhm_right": int}.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
    peak_colors : dict[str, str], optional
        Colors per label (e.g., {"P":"orange","Q":"green",...}). Defaults are used if None.
    title : str, default "Detected Peaks and FWHM Boundaries"
        Plot title.
    xlabel : str, default "Cycle-Relative Index"
        X label.
    ylabel : str, default "Amplitude"
        Y label.
    pad_y : float, default 0.1
        Vertical padding added to ylim as +/- `pad_y` * signal range.
    show : bool, default True
        If True, calls plt.show() at the end.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.
    """
    xs = np.asarray(xs)
    signal = np.asarray(signal, dtype=float)

    if xs.ndim != 1 or signal.ndim != 1 or xs.size != signal.size:
        raise ValueError("`xs` and `signal` must be 1D arrays of equal length.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Default peak colors
    if peak_colors is None:
        peak_colors = {"P": "orange", "Q": "green", "R": "red", "S": "purple", "T": "magenta"}

    # Base signal
    ax.plot(xs, signal, label="Detrended Signal", linewidth=2)

    # Track whether we've added half-max labels yet (to avoid duplicates in legend)
    left_labeled = False
    right_labeled = False

    n = signal.size
    for label, center_idx in peak_inds.items():
        if center_idx is None or not (0 <= center_idx < n):
            continue

        color = peak_colors.get(label, "black")

        # Center marker
        ax.scatter(
            xs[center_idx],
            signal[center_idx],
            s=80,
            edgecolor="white",
            zorder=3,
            label=f"{label} Peak",
            c=[color],
        )

        # FWHM markers/line if present
        fwhm = fwhm_results.get(label, {})
        left = fwhm.get("fwhm_left")
        right = fwhm.get("fwhm_right")

        if left is not None and right is not None and 0 <= left < n and 0 <= right < n:
            # Left half-max
            ax.scatter(
                xs[left],
                signal[left],
                marker="x",
                s=100,
                zorder=2,
                label=None if left_labeled else "Left Half-Max",
            )
            left_labeled = True

            # Right half-max
            ax.scatter(
                xs[right],
                signal[right],
                marker="x",
                s=100,
                zorder=2,
                label=None if right_labeled else "Right Half-Max",
            )
            right_labeled = True

            # Line between half-max points
            ax.plot(
                [xs[left], xs[right]],
                [signal[left], signal[right]],
                linestyle="dotted",
                linewidth=1.5,
                alpha=0.7,
                color=color,
            )

    # Labels & layout
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Deduplicate while preserving order
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        if uniq:
            ax.legend(*zip(*uniq), fontsize=11, loc="best")

    # Y-limits with padding
    y_min, y_max = np.nanmin(signal), np.nanmax(signal)
    if np.isfinite(y_min) and np.isfinite(y_max):
        rng = max(y_max - y_min, np.finfo(float).eps)
        ax.set_ylim(y_min - pad_y * rng, y_max + pad_y * rng)

    plt.tight_layout()
    if show:
        plt.show()

    return ax
