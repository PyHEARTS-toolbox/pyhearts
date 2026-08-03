from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.signal import find_peaks

from pyhearts._morphology import plts as _morph_plts
from pyhearts._morphology.config import ProcessCycleConfig


def detect_signal_polarity(
    ecg: np.ndarray,
    sampling_rate: float,
    *,
    min_refrac_ms: float = 100.0,
) -> bool:
    """
    Detect whether QRS complexes are inverted (dominant negative R/nadirs).

    Compares positive vs negative extrema with a conservative bias toward
    *normal* (non-inverted) when the evidence is ambiguous. Amplitude checks
    are scale-free (MAD-based), so unitless synthetic traces and mV recordings
    both work.

    Returns
    -------
    bool
        ``True`` if the signal should be polarity-flipped for R detection /
        morphology; ``False`` otherwise.
    """
    ecg = np.asarray(ecg, dtype=float)
    if ecg.ndim != 1 or ecg.size < 100:
        return False
    if sampling_rate <= 0:
        raise ValueError("`sampling_rate` must be > 0.")

    mad = float(np.median(np.abs(ecg - np.median(ecg))))
    robust_std = 1.4826 * mad if mad > 1e-9 else float(np.std(ecg))
    if not np.isfinite(robust_std) or robust_std <= 0:
        return False

    prominence_threshold = 2.0 * robust_std
    distance_samples = max(1, int(round(min_refrac_ms * sampling_rate / 1000.0)))
    # "Substantial" deflection relative to noise (replaces hardcoded 0.5 mV).
    amp_floor = max(3.0 * robust_std, 1e-9)

    pos_peaks, pos_props = find_peaks(
        ecg, distance=distance_samples, prominence=prominence_threshold
    )
    neg_peaks, neg_props = find_peaks(
        -ecg, distance=distance_samples, prominence=prominence_threshold
    )

    if pos_peaks.size < 3 and neg_peaks.size < 3:
        relaxed = prominence_threshold * 0.6
        pos_peaks, pos_props = find_peaks(
            ecg, distance=distance_samples, prominence=relaxed
        )
        neg_peaks, neg_props = find_peaks(
            -ecg, distance=distance_samples, prominence=relaxed
        )
        if pos_peaks.size < 2 and neg_peaks.size < 2:
            return False

    max_pos_val = float(np.max(ecg[pos_peaks])) if pos_peaks.size else 0.0
    min_neg_val = float(np.min(ecg[neg_peaks])) if neg_peaks.size else 0.0

    # Primary: compare actual extremal amplitudes (conservative toward normal).
    if max_pos_val > amp_floor and max_pos_val > abs(min_neg_val) * 0.7:
        return False
    if abs(min_neg_val) > amp_floor and abs(min_neg_val) > max_pos_val * 1.3:
        return True

    if pos_peaks.size > 0 and neg_peaks.size > 0:
        pos_prom = pos_props.get("prominences")
        neg_prom = neg_props.get("prominences")
        if pos_prom is None:
            pos_prom = ecg[pos_peaks] - np.median(ecg)
            pos_prom = pos_prom[pos_prom > 0]
        if neg_prom is None:
            neg_prom = np.median(ecg) - ecg[neg_peaks]
            neg_prom = neg_prom[neg_prom > 0]

        median_pos_prom = float(np.median(pos_prom)) if len(pos_prom) else 0.0
        median_neg_prom = float(np.median(neg_prom)) if len(neg_prom) else 0.0
        median_pos_amp = float(np.median(np.abs(ecg[pos_peaks])))
        median_neg_amp = float(np.median(np.abs(ecg[neg_peaks])))

        if median_neg_prom > 1.3 * median_pos_prom and median_pos_prom > 0:
            return True
        if median_neg_amp > 1.4 * median_pos_amp and median_pos_amp > 0:
            return True
        return False

    if neg_peaks.size > 0 and pos_peaks.size == 0:
        return True
    return False


def r_peak_detection(
    ecg: Union[np.ndarray, list[float]],
    sampling_rate: float,
    *,
    cfg: ProcessCycleConfig,
    plot: bool = False,
    plot_start: Optional[float] = None,  # seconds
    plot_end: Optional[float] = None,  # seconds
    crop_ms: Optional[int] = 3000,  # plotting convenience only
    auto_polarity: Optional[bool] = None,
) -> np.ndarray:
    """
    Two-pass, prominence-based R-peak detection driven by config.

    First-pass uses a fixed refractory (``cfg.rpeak_min_refrac_ms``) to estimate
    RR. Second-pass uses ``cfg.rpeak_rr_frac_second_pass * median(RR)`` as
    refractory. RR estimate is clamped using ``cfg.rpeak_bpm_bounds``.

    When auto-polarity is enabled (default via ``cfg.rpeak_auto_polarity``),
    inverted QRS leads are detected and peak-finding runs on the negated
    trace so nadirs are recovered as R indices on the original sample axis.
    """
    ecg = np.asarray(ecg, dtype=float)
    if ecg.ndim != 1 or ecg.size == 0:
        raise ValueError("`ecg` must be a non-empty 1D array.")
    if sampling_rate <= 0:
        raise ValueError("`sampling_rate` must be > 0.")

    use_auto = cfg.rpeak_auto_polarity if auto_polarity is None else bool(auto_polarity)
    is_inverted = (
        detect_signal_polarity(
            ecg,
            sampling_rate,
            min_refrac_ms=cfg.rpeak_min_refrac_ms,
        )
        if use_auto
        else False
    )
    ecg_for_peaks = -ecg if is_inverted else ecg

    # ----- First pass -----
    distance_lo = max(1, int(round(cfg.rpeak_min_refrac_ms * sampling_rate / 1000.0)))
    prominence_threshold = cfg.rpeak_prominence_multiplier * float(np.std(ecg_for_peaks))
    peaks_lo, _ = find_peaks(
        ecg_for_peaks, distance=distance_lo, prominence=prominence_threshold
    )

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
            ecg_for_peaks, distance=distance, prominence=prominence_threshold
        )

    final_filtered_r_peaks = np.asarray(initial_r_peaks, dtype=int)

    # ----- Optional plotting (original polarity) -----
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

            _morph_plts.plot_rpeaks(
                ecg_segment,
                sampling_rate,
                peaks_in_window,
                crop_ms=crop_ms,
                title="ECG (windowed) with R-peaks",
            )
        else:
            _morph_plts.plot_rpeaks(
                ecg,
                sampling_rate,
                final_filtered_r_peaks,
                crop_ms=crop_ms,
                title="ECG with R-peaks",
            )

    return final_filtered_r_peaks
