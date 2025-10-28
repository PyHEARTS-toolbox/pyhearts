from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks
import pywt
from pyhearts.config import ProcessCycleConfig   


from pyhearts.plts import plot_epochs



def epoch_ecg(
    ecg: Union[np.ndarray, List[float]],
    r_peaks: Union[np.ndarray, List[int]],
    sampling_rate: float,
    *,
    cfg: Optional[ProcessCycleConfig] = None,
    verbose: bool = False,
    plot: bool = False,
    corr_thresh: Optional[float] = None,
    var_thresh: Optional[float] = None,
    estimate_energy: bool = False,
    wavelet_name: str = "db6",
) -> Tuple[pd.DataFrame, Optional[float]]:

    """
    Epoch an ECG signal around R-peaks, filter noisy cycles, and (optionally) estimate
    a wavelet-based energy threshold.

    Parameters
    ----------
    ecg : array-like
        1D ECG signal.
    r_peaks : array-like
        Indices (samples) of detected R-peaks.
    sampling_rate : float
        Sampling rate in Hz.
    verbose : bool, default=False
        If True, print diagnostic information.
    plot : bool, default=False
        If True, plot retained epochs using `plot_epochs`.
    corr_thresh : float, default=0.8
        Minimum correlation with the global template to keep a cycle.
    var_thresh : float, default=5.0
        Maximum allowed multiple of the global signal variance for a cycle.
    estimate_energy : bool, default=False
        If True, compute and return the 95th percentile of a wavelet-based
        energy proxy across retained cycles.
    wavelet_name : str, default="db6"
        PyWavelets wavelet name for energy estimation.

    Returns
    -------
    epochs_df : pandas.DataFrame
        Long-form table with columns:
        ['signal_x', 'signal_y', 'index', 'cycle'].
    expected_max_energy : float or None
        95th percentile of wavelet detail-coefficient energy across valid cycles
        if `estimate_energy=True`, otherwise None.

    Notes
    -----
    - Cycles are extracted as symmetric windows around each R-peak with half-width
      equal to half the average RR interval.
    - Cycles are retained if they meet both a correlation (to the global template)
      and a variance criterion relative to the whole signal.
    """
    # ---- config ----
    if cfg is None:
        cfg = ProcessCycleConfig()
    # allow per-call overrides; otherwise use cfg defaults
    corr_thresh = cfg.epoch_corr_thresh if corr_thresh is None else float(corr_thresh)
    var_thresh  = cfg.epoch_var_thresh  if var_thresh  is None else float(var_thresh)

    # --- Convert inputs ---
    ecg = np.asarray(ecg, dtype=float)
    r_peaks = np.asarray(r_peaks, dtype=int)

    # --- Basic validation ---
    if ecg.ndim != 1:
        raise ValueError("`ecg` must be a 1D array.")
    if r_peaks.ndim != 1:
        raise ValueError("`r_peaks` must be a 1D array of sample indices.")
    if r_peaks.size < 2:
        # Need at least 2 R-peaks to estimate a window from RR intervals
        return pd.DataFrame(columns=["signal_x", "signal_y", "index", "cycle"]), None

    # --- Window size from RR ---
    rr_intervals_samples = np.diff(r_peaks)
    avg_rr_interval = float(np.mean(rr_intervals_samples))

    # use explicit None-check so 0 is a valid fixed window if you ever want that
    pre_r = (
        cfg.pre_r_window
        if cfg.pre_r_window is not None
        else int(round(avg_rr_interval / 2))
    )
    cfg = cfg or ProcessCycleConfig()  # safe default
    
    # --- Convert inputs ---
    ecg = np.asarray(ecg, dtype=float)
    r_peaks = np.asarray(r_peaks, dtype=int)

    # --- Basic validation ---
    if ecg.ndim != 1:
        raise ValueError("`ecg` must be a 1D array.")
    if r_peaks.ndim != 1:
        raise ValueError("`r_peaks` must be a 1D array of sample indices.")
    if r_peaks.size < 2:
        # Need at least 2 R-peaks to estimate a window from RR intervals
        return pd.DataFrame(columns=["signal_x", "signal_y", "index", "cycle"]), None

    # --- Window size from RR ---
    rr_intervals_samples = np.diff(r_peaks)
    avg_rr_interval = float(np.mean(rr_intervals_samples))
    pre_r = cfg.pre_r_window if cfg.pre_r_window is not None else int(round(avg_rr_interval / 2))

    if pre_r <= 1:
        # Degenerate window; return empty result
        return pd.DataFrame(columns=["signal_x", "signal_y", "index", "cycle"]), None

    ecg_indices = np.arange(ecg.size)

    all_cycles: List[np.ndarray] = []
    all_metadata: List[dict] = []

    # --- Extract windows around each R-peak ---
    for idx, r_peak in enumerate(r_peaks):
        start = r_peak - pre_r
        end = r_peak + pre_r
        if start < 0 or end > ecg.size:
            continue  # skip incomplete windows at boundaries

        window = ecg[start:end]
        if window.size != 2 * pre_r:
            continue

        window_detrended = detrend(window, type="linear")

        x_vals = np.linspace(-pre_r / sampling_rate, pre_r / sampling_rate, window_detrended.size)
        r_peak_lat = x_vals[int(np.argmax(window_detrended))]

        all_cycles.append(window_detrended)
        all_metadata.append(
            {"idx": idx, "start": start, "x_vals": x_vals, "r_peak_lat": r_peak_lat}
        )

    if not all_cycles:
        if verbose:
            print("No complete cycles extracted.")
        return pd.DataFrame(columns=["signal_x", "signal_y", "index", "cycle"]), None

    all_cycles_arr = np.vstack(all_cycles)
    global_template = np.mean(all_cycles_arr, axis=0)
    ecg_var = float(np.var(ecg))

    epochs_rows: List[dict] = []
    kept_cycles: List[np.ndarray] = []
    kept_metadata: List[dict] = []
    r_latencies: List[float] = []  # kept for possible downstream use

    # --- Filter cycles by correlation and variance ---
    for cycle, meta in zip(all_cycles_arr, all_metadata):
        corr = np.corrcoef(cycle, global_template)[0, 1]
        if np.isnan(corr) or corr < corr_thresh:
            continue
        if np.var(cycle) > var_thresh * ecg_var:
            continue

        # Append rows for this retained cycle (long-form table)
        start = meta["start"]
        x_vals = meta["x_vals"]
        for i, y in enumerate(cycle):
            epochs_rows.append(
                {
                    "signal_x": x_vals[i],
                    "signal_y": float(y),
                    "index": int(ecg_indices[start + i]),
                    "cycle": int(meta["idx"]),
                }
            )

        kept_cycles.append(cycle)
        kept_metadata.append(meta)
        r_latencies.append(float(meta["r_peak_lat"]))

    if verbose:
        print(f"Total R-peaks: {r_peaks.size}")
        print(f"Valid cycles after filtering: {len(kept_cycles)}")

    if not kept_cycles:
        # Nothing survived filtering
        if plot:
            # nothing to plot
            pass
        epochs_df = pd.DataFrame(columns=["signal_x", "signal_y", "index", "cycle"])
        return (epochs_df, None) if not estimate_energy else (epochs_df, 1.0)

    # --- Plot (use the x-axis from the first kept cycle) ---
    if plot:
        plot_epochs(kept_cycles, kept_metadata[0]["x_vals"])

    epochs_df = pd.DataFrame(epochs_rows, columns=["signal_x", "signal_y", "index", "cycle"])

    # --- Optional wavelet-based energy estimate ---
    expected_max_energy: Optional[float] = None
    if estimate_energy:
        energies: List[float] = []
        for sig in kept_cycles:
            coeffs = pywt.wavedec(sig, wavelet_name, level=3)
            # Heuristic: use a mid-level detail band as QRS proxy
            detail_coeffs = coeffs[2] if len(coeffs) > 2 else coeffs[-2]
            peaks, _ = find_peaks(np.abs(detail_coeffs), height=np.std(detail_coeffs) * 1.2)
            if peaks.size > 0:
                qrs_energy = float(np.sum(np.abs(detail_coeffs[peaks])) / peaks.size)
                energies.append(qrs_energy)

        expected_max_energy = float(np.percentile(energies, 95)) if energies else 1.0

    return epochs_df, expected_max_energy
