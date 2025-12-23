import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import Optional

from pyhearts.processing.rpeak import _detect_signal_polarity


def _bandpass_5_15(ecg: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Narrow bandpass used prior to derivative-based R-peak energy detection."""
    nyq = sampling_rate / 2.0
    low = max(0.01, 5.0 / nyq)
    high = max(low + 0.01, min(15.0 / nyq, 0.99))
    b, a = butter(2, [low, high], btype="band")
    return filtfilt(b, a, ecg)


def _moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    if win == 1:
        return x
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")


def bandpass_energy_r_peak_detection(
    ecg: np.ndarray,
    sampling_rate: float,
    cfg,
    plot: bool = False,
    raw_ecg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    R-peak detection based on narrow band-pass filtering and derivative energy.

    Steps:
    1) Bandpass 5–15 Hz to emphasize QRS complexes.
    2) Derivative + squared energy, 150 ms moving-average integration.
    3) Initial threshold from the first 3 s (signal vs. noise peaks).
       umb1 = noise + 0.25*(signal - noise).
    4) Peak search on the integrated energy with minimum distance set by
       cfg.rpeak_min_refrac_ms.
    5) Map each energy peak back to the filtered signal as the maximum absolute
       amplitude within ±50 ms.
    6) Optional BPM-guard using cfg.rpeak_bpm_bounds.
    """
    ecg = np.asarray(ecg, dtype=float)
    if ecg.ndim != 1 or ecg.size == 0:
        raise ValueError("`ecg` must be a non-empty 1D array.")
    if sampling_rate <= 0:
        raise ValueError("`sampling_rate` must be > 0.")

    # Polarity detection on raw (preferred) or filtered signal
    polarity_signal = raw_ecg if raw_ecg is not None else ecg
    is_inverted = _detect_signal_polarity(
        polarity_signal, sampling_rate, min_refrac_ms=cfg.rpeak_min_refrac_ms
    )

    filtered = _bandpass_5_15(ecg, sampling_rate)
    if is_inverted:
        filtered = -filtered

    # Derivative energy + 150 ms moving average
    derivative = np.gradient(filtered)
    energy = derivative ** 2
    win_samples = int(round(0.150 * sampling_rate))
    integrated = _moving_avg(energy, win_samples)

    # Training region: first 3 seconds (or all if shorter)
    train_end = min(len(integrated), int(round(3.0 * sampling_rate)))
    train_seg = integrated[:train_end] if train_end > 0 else integrated
    if train_seg.size == 0:
        return np.array([], dtype=int)

    signal_peak = float(np.max(train_seg))
    below_cut = train_seg[train_seg < 0.75 * signal_peak]
    noise_peak = float(np.median(below_cut)) if below_cut.size > 0 else 0.5 * signal_peak
    umb1 = noise_peak + 0.25 * (signal_peak - noise_peak)
    min_distance = max(1, int(round(cfg.rpeak_min_refrac_ms * sampling_rate / 1000.0)))

    # Find peaks on integrated energy
    peaks_energy, _ = find_peaks(integrated, height=umb1, distance=min_distance)
    if peaks_energy.size == 0:
        return np.array([], dtype=int)

    # Map integrated peaks to actual R positions on filtered signal
    search_radius = int(round(0.050 * sampling_rate))  # ±50 ms
    r_peaks = []
    for p in peaks_energy:
        start = max(0, p - search_radius)
        end = min(len(filtered), p + search_radius)
        if end <= start:
            continue
        local = filtered[start:end]
        local_idx = int(np.argmax(np.abs(local)))
        r_peaks.append(start + local_idx)

    if not r_peaks:
        return np.array([], dtype=int)

    r_peaks = np.unique(np.array(r_peaks, dtype=int))

    # Refractory enforcement (sorted unique already)
    kept = [r_peaks[0]]
    for rp in r_peaks[1:]:
        if rp - kept[-1] >= min_distance:
            kept.append(rp)
    r_peaks = np.array(kept, dtype=int)

    # Optional BPM bounds filtering
    if r_peaks.size > 1:
        rr_ms = np.diff(r_peaks) / sampling_rate * 1000.0
        bpm = 60000.0 / rr_ms
        lo_bpm, hi_bpm = cfg.rpeak_bpm_bounds
        valid = np.ones_like(r_peaks, dtype=bool)
        for i, b in enumerate(bpm):
            if b < lo_bpm or b > hi_bpm:
                if i + 1 < valid.size:
                    valid[i + 1] = False
        r_peaks = r_peaks[valid]

    if plot:
        try:
            import pyhearts as ph

            ph.plots.plot_rpeaks(
                ecg,
                sampling_rate,
                r_peaks,
                crop_ms=3000,
                title="ECG with R-peaks (bandpass-energy method)",
            )
        except Exception:
            # plotting is optional; ignore failures
            pass

    return r_peaks


