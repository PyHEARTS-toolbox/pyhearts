import numpy as np
import pyhearts as ph
from scipy.signal import find_peaks
from typing import Optional, Union
from pyhearts.config import ProcessCycleConfig


def r_peak_detection(
    ecg: Union[np.ndarray, list[float]],
    sampling_rate: float,
    *,
    cfg: ProcessCycleConfig,
    plot: bool = False,
    plot_start: Optional[float] = None,   # seconds
    plot_end: Optional[float] = None,     # seconds
    crop_ms: Optional[int] = 3000,        # plotting convenience only
) -> np.ndarray:
    """
    Two-pass, prominence-based R-peak detection driven by config.
    First-pass uses a fixed refractory (cfg.rpeak_min_refrac_ms) to estimate RR.
    Second-pass uses cfg.rpeak_rr_frac_second_pass * median(RR) as refractory.
    RR estimate is clamped using cfg.rpeak_bpm_bounds.
    """
    ecg = np.asarray(ecg, dtype=float)
    if ecg.ndim != 1 or ecg.size == 0:
        raise ValueError("`ecg` must be a non-empty 1D array.")
    if sampling_rate <= 0:
        raise ValueError("`sampling_rate` must be > 0.")

    # ----- First pass -----
    distance_lo = max(1, int(round(cfg.rpeak_min_refrac_ms * sampling_rate / 1000.0)))
    prominence_threshold = cfg.rpeak_prominence_multiplier * float(np.std(ecg))
    peaks_lo, _ = find_peaks(ecg, distance=distance_lo, prominence=prominence_threshold)

    # ----- RR estimation and second pass -----
    if peaks_lo.size < 3:
        initial_r_peaks = peaks_lo.astype(int)
    else:
        rr_samp = np.median(np.diff(peaks_lo))

        min_bpm, max_bpm = cfg.rpeak_bpm_bounds
        rr_min = (60_000.0 / max_bpm) * sampling_rate / 1000.0
        rr_max = (60_000.0 / min_bpm) * sampling_rate / 1000.0
        rr_samp = float(np.clip(rr_samp, rr_min, rr_max))

        distance = max(1, int(round(cfg.rpeak_rr_frac_second_pass * rr_samp)))
        initial_r_peaks, _ = find_peaks(
            ecg, distance=distance, prominence=prominence_threshold
        )

    final_filtered_r_peaks = np.asarray(initial_r_peaks, dtype=int)

    # ----- Optional plotting -----
    if plot:
        if plot_start is not None and plot_end is not None:
            if plot_start < 0 or plot_end <= plot_start:
                raise ValueError("`plot_end` must be > `plot_start` and both ≥ 0.")
            start_idx = int(round(plot_start * sampling_rate))
            end_idx = int(round(plot_end * sampling_rate))
            start_idx = max(0, min(start_idx, ecg.size))
            end_idx = max(start_idx + 1, min(end_idx, ecg.size))

            ecg_segment = ecg[start_idx:end_idx]
            peaks_in_window = final_filtered_r_peaks[
                (final_filtered_r_peaks >= start_idx) & (final_filtered_r_peaks < end_idx)
            ] - start_idx

            ph.plots.plot_rpeaks(
                ecg_segment,
                sampling_rate,
                peaks_in_window,
                crop_ms=crop_ms,
                title="ECG (windowed) with R-peaks",
            )
        else:
            ph.plots.plot_rpeaks(
                ecg,
                sampling_rate,
                final_filtered_r_peaks,
                crop_ms=crop_ms,
                title="ECG with R-peaks",
            )

    return final_filtered_r_peaks
