from typing import Dict, Optional, Union, Mapping
import numpy as np
import matplotlib.pyplot as plt

def plot_rise_decay(
    xs: Union[np.ndarray, list[float]],
    sig: Union[np.ndarray, list[float]],
    peak_data: Dict[str, Mapping[str, Optional[float]]],
    *,
    ax: Optional[plt.Axes] = None,
    peak_colors: Optional[Dict[str, str]] = None,
    title: str = "Rise and Decay Times for Selected Cycle",
    xlabel: str = "Time (ms)",
    ylabel: str = "Amplitude",
    label_offset: float = 0.05,
    pad_y: float = 0.1,
    line_width: float = 2.0,
    show: bool = True,
) -> plt.Axes:
    """
    Plot rise (left→center) and decay (center→right) segments for detected ECG peaks.

    Parameters
    ----------
    xs : array-like
        1D x-axis values (time or sample index).
    sig : array-like
        1D ECG signal values aligned with `xs`.
    peak_data : dict[str, Mapping[str, Optional[float]]]
        For each component label (e.g., "P","Q","R","S","T"), a mapping with:
        - "le_idx": left-bound index (int-like)
        - "center_idx": center index (int-like)
        - "ri_idx": right-bound index (int-like)
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure/axes is created.
    peak_colors : dict[str, str], optional
        Colors per label, e.g., {"P":"orange","Q":"green",...}.
    title : str, default "Rise and Decay Times for Selected Cycle"
        Plot title.
    xlabel : str, default "Time (ms)"
        X-axis label.
    ylabel : str, default "Amplitude"
        Y-axis label.
    label_offset : float, default 0.05
        Vertical offset (signal units) for component text labels.
    pad_y : float, default 0.1
        Fractional padding added to y-limits.
    line_width : float, default 2.0
        Line width for rise/decay segments.
    show : bool, default True
        If True, calls `plt.show()` at the end.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.
    """
    xs = np.asarray(xs)
    sig = np.asarray(sig, dtype=float)

    if xs.ndim != 1 or sig.ndim != 1 or xs.size != sig.size:
        raise ValueError("`xs` and `sig` must be 1D arrays of equal length.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if peak_colors is None:
        peak_colors = {"P": "orange", "Q": "green", "R": "red", "S": "purple", "T": "magenta"}

    # Base signal
    ax.plot(xs, sig, color="dodgerblue", linewidth=2, label="Detrended Signal", zorder=1)

    n = sig.size
    for comp, pdata in peak_data.items():
        color = peak_colors.get(comp, "black")

        le = pdata.get("le_idx")
        c  = pdata.get("center_idx")
        ri = pdata.get("ri_idx")

        # Validate presence and finiteness
        if le is None or c is None or ri is None:
            continue
        if not np.isfinite(le) or not np.isfinite(c) or not np.isfinite(ri):
            continue

        # Cast to int and bounds-check
        le_i, c_i, ri_i = int(le), int(c), int(ri)
        if not (0 <= le_i < n and 0 <= c_i < n and 0 <= ri_i < n):
            continue

        # Points
        le_x, le_y = xs[le_i], sig[le_i]
        c_x,  c_y  = xs[c_i],  sig[c_i]
        ri_x, ri_y = xs[ri_i], sig[ri_i]

        # Rise (left -> center)
        ax.plot([le_x, c_x], [le_y, c_y], linestyle="-", linewidth=line_width,
                color=color, label=f"{comp} Rise", zorder=2)
        # Decay (center -> right)
        ax.plot([c_x, ri_x], [c_y, ri_y], linestyle="--", linewidth=line_width,
                color=color, label=f"{comp} Decay", zorder=2)

        # Center marker + label
        ax.scatter(c_x, c_y, color=color, edgecolor="white", zorder=3, s=60, label=f"{comp} Peak")
        ax.text(c_x, c_y + (label_offset if c_y >= 0 else -label_offset),
                comp, color=color, fontsize=12, ha="center", fontweight="bold", zorder=4)

    # Labels, legend, limits
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h); uniq_l.append(l); seen.add(l)
    if uniq_h:
        ax.legend(uniq_h, uniq_l, fontsize=11, loc="best")

    # Y padding
    y_min, y_max = np.nanmin(sig), np.nanmax(sig)
    rng = max(y_max - y_min, np.finfo(float).eps)
    ax.set_ylim(y_min - pad_y * rng, y_max + pad_y * rng)

    plt.tight_layout()
    if show:
        plt.show()

    return ax
