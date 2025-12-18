import numpy as np
import pyhearts as ph
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from typing import Optional, Union
from pyhearts.config import ProcessCycleConfig


def _bandpass_filter(
    ecg: np.ndarray,
    sampling_rate: float,
    highpass_hz: float,
    lowpass_hz: float,
    order: int = 2,
    notch_hz: Optional[float] = None,
) -> np.ndarray:
    """Apply bandpass (and optional notch) filter for R-peak detection."""
    filtered = ecg.copy()
    
    # Highpass to remove baseline wander
    if highpass_hz > 0:
        nyq = sampling_rate / 2
        # Ensure cutoff is valid
        hp_norm = min(highpass_hz / nyq, 0.99)
        b, a = butter(order, hp_norm, btype='high')
        filtered = filtfilt(b, a, filtered)
    
    # Lowpass to remove high-frequency noise
    if lowpass_hz > 0:
        nyq = sampling_rate / 2
        lp_norm = min(lowpass_hz / nyq, 0.99)
        b, a = butter(order, lp_norm, btype='low')
        filtered = filtfilt(b, a, filtered)
    
    # Optional notch filter for power line interference
    if notch_hz is not None and notch_hz > 0:
        q = 30.0  # Quality factor
        b, a = iirnotch(notch_hz, q, sampling_rate)
        filtered = filtfilt(b, a, filtered)
    
    return filtered


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
    
    Optionally applies bandpass filtering (cfg.rpeak_preprocess) for noise robustness.
    First-pass uses a fixed refractory (cfg.rpeak_min_refrac_ms) to estimate RR.
    Second-pass uses cfg.rpeak_rr_frac_second_pass * median(RR) as refractory.
    RR estimate is clamped using cfg.rpeak_bpm_bounds.
    """
    ecg = np.asarray(ecg, dtype=float)
    if ecg.ndim != 1 or ecg.size == 0:
        raise ValueError("`ecg` must be a non-empty 1D array.")
    if sampling_rate <= 0:
        raise ValueError("`sampling_rate` must be > 0.")

    # ----- Optional preprocessing for noise robustness -----
    if cfg.rpeak_preprocess:
        ecg_filtered = _bandpass_filter(
            ecg,
            sampling_rate,
            highpass_hz=cfg.rpeak_highpass_hz,
            lowpass_hz=cfg.rpeak_lowpass_hz,
            order=cfg.rpeak_filter_order,
            notch_hz=cfg.rpeak_notch_hz,
        )
    else:
        ecg_filtered = ecg

    # ----- First pass -----
    distance_lo = max(1, int(round(cfg.rpeak_min_refrac_ms * sampling_rate / 1000.0)))
    prominence_threshold = cfg.rpeak_prominence_multiplier * float(np.std(ecg_filtered))
    peaks_lo, _ = find_peaks(ecg_filtered, distance=distance_lo, prominence=prominence_threshold)

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
            ecg_filtered, distance=distance, prominence=prominence_threshold
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
