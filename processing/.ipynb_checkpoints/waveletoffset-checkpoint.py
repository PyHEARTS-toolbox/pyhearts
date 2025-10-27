from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
from scipy.signal import find_peaks
import pywt
import pyhearts.plts
from pyhearts.config import ProcessCycleConfig


def calc_wavelet_dynamic_offset(
    ecg_signal: Union[np.ndarray, list[float]],
    sampling_rate: float,
    expected_max_energy: float,
    *,
    xs: Optional[Union[np.ndarray, list[float]]] = None,
    r_center_idx: Optional[int] = None,
    r_std: Optional[float] = None,
    plot: bool = False,
    cfg: Optional[ProcessCycleConfig] = None,
) -> Tuple[int, Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Compute a dynamic offset (in samples) scaled by wavelet-based QRS energy,
    and optionally return R/Q/S bounds for plotting context.

    Parameters
    ----------
    ecg_signal : array-like
        1D ECG signal.
    sampling_rate : float
        Sampling rate in Hz (> 0).
    expected_max_energy : float
        Reference wavelet energy used for scaling (must be > 0 to have effect).
    xs : array-like, optional
        X-axis (time) vector; required for plotting.
    r_center_idx : int, optional
        R-peak center index; required for bound computation/plotting context.
    r_std : float, optional
        Standard deviation around the R-peak for bound computation (must be > 0).
    plot : bool, default False
        If True, plot dynamic offset bounds when sufficient context is provided.
    cfg : ProcessCycleConfig, optional
        Configuration with wavelet/offset parameters.

    Returns
    -------
    dynamic_offset_samples : int
        Offset in samples, clamped within the ms range then converted to samples.
    r_left_idx, r_right_idx, q_min_idx, s_max_idx : Optional[int]
        Derived indices for plotting context if `r_center_idx` and `r_std` are provided, else None.
    """
    # --- Config & input coercion ---
    cfg = cfg or ProcessCycleConfig()

    # ---- Coerce/validate inputs ----
    sig = np.asarray(ecg_signal, dtype=float)
    if sig.ndim != 1 or sig.size == 0 or sampling_rate <= 0:
        default = int(round(sampling_rate * (cfg.wavelet_base_offset_ms / 1000.0))) if sampling_rate > 0 else 0
        return default, None, None, None, None

    # Offsets in ms → samples
    lo_ms = float(cfg.wavelet_base_offset_ms)
    hi_ms = float(cfg.wavelet_max_offset_ms)
    if lo_ms > hi_ms:
        lo_ms, hi_ms = hi_ms, lo_ms
    default_offset_samples = int(round(sampling_rate * (lo_ms / 1000.0)))

    ref_energy = float(expected_max_energy) if expected_max_energy > 0 else 1.0

    # ---- Wavelet decomposition (choose requested detail level, clamp to available) ----
    try:
        w = pywt.Wavelet(cfg.wavelet_name)
        # Use detail at the requested level if available, else fall back to the highest possible.
        max_level = pywt.dwt_max_level(sig.size, w.dec_len)
        n = max(1, min(cfg.wavelet_detail_level, max_level))
        coeffs = pywt.wavedec(sig, cfg.wavelet_name, level=n)
        detail = coeffs[1]  # detail at level n

    except Exception:
        return default_offset_samples, None, None, None, None

    # ---- Peak detection on wavelet detail coefficients ----
    if detail.size < 3 or np.allclose(detail, 0):
        return default_offset_samples, None, None, None, None

    height_thresh = float(np.std(detail) * float(cfg.wavelet_peak_height_sigma))
    peaks, _ = find_peaks(np.abs(detail), height=height_thresh)
    if peaks.size == 0:
        return default_offset_samples, None, None, None, None

    # ---- Energy scaling → dynamic offset (ms → samples) ----
    qrs_energy = float(np.sum(np.abs(detail[peaks])) / peaks.size)
    relative_energy = float(np.clip(qrs_energy / ref_energy, 0.0, 1.0))
    dynamic_offset_ms = lo_ms + (hi_ms - lo_ms) * relative_energy
    dynamic_offset_samples = int(round(sampling_rate * (dynamic_offset_ms / 1000.0)))

    # ---- Optional R/Q/S bounds and plotting ----
    r_left_idx = r_right_idx = q_min_idx = s_max_idx = None
    if r_center_idx is not None and r_std is not None and r_std > 0:
        k = float(cfg.wavelet_k_multiplier)
        r_left_idx = int(round(r_center_idx - k * r_std))
        r_right_idx = int(round(r_center_idx + k * r_std))
        r_left_idx = max(0, min(r_left_idx, sig.size - 1))
        r_right_idx = max(0, min(r_right_idx, sig.size - 1))

        q_min_idx = max(0, r_left_idx - dynamic_offset_samples)
        s_max_idx = min(sig.size - 1, r_right_idx + dynamic_offset_samples)

        if plot and xs is not None:
            xs_arr = np.asarray(xs)
            if xs_arr.shape == sig.shape:
                pyhearts.plts.plot_dynamic_offset(
                    xs=xs_arr,
                    sig=sig,
                    r_center_idx=int(r_center_idx),
                    r_left_idx=r_left_idx,
                    r_right_idx=r_right_idx,
                    q_min_idx=q_min_idx,
                    s_max_idx=s_max_idx,
                )

    return dynamic_offset_samples, r_left_idx, r_right_idx, q_min_idx, s_max_idx
    
