"""
Local Q- and S-wave extremum search for record-level STPQ template anchoring.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig

# Physiological search margins (aligned with Step-3 QTDB anchor diagnostic).
_QS_INNER_MARGIN_MS = 5.0
_S_DOWNSTROKE_CHECK_MS = 20.0
_S_DOWNSTROKE_RETRY_LO_MS = 40.0


def qs_search_window_ms(cfg: ProcessCycleConfig) -> float:
    """Legacy alias: Q search extent before R (ms)."""
    return cfg.record_q_search_before_r_ms


def s_search_after_r_ms(cfg: ProcessCycleConfig) -> float:
    """S search extent after R (ms)."""
    return cfg.record_s_search_after_r_ms


def q_search_before_r_ms(cfg: ProcessCycleConfig) -> float:
    """Q search extent before R (ms)."""
    return cfg.record_q_search_before_r_ms


def _ms_to_samples(ms: float, sampling_rate: float) -> int:
    return max(1, int(round(ms * sampling_rate / 1000.0)))


def _segment_argext(
    segment: np.ndarray,
    *,
    inverted: bool,
) -> int:
    return int(np.argmax(segment)) if inverted else int(np.argmin(segment))


def _is_still_descending(
    ecg: np.ndarray,
    idx: int,
    sampling_rate: float,
) -> bool:
    """True when ``idx`` is on the QRS downstroke (deeper trough likely later)."""
    delta = _ms_to_samples(_S_DOWNSTROKE_CHECK_MS, sampling_rate)
    j = int(idx) + delta
    if j >= len(ecg):
        return False
    return float(ecg[int(idx)]) > float(ecg[j])


def find_q_wave_before_r(
    ecg: np.ndarray,
    r_idx: int,
    sampling_rate: float,
    *,
    search_window_ms: float,
    inverted: bool = False,
) -> Optional[int]:
    """
    Deepest Q trough before R within ``[R − window, R − inner_margin]``.

    Excludes samples immediately adjacent to R so the pre-R upswing is not mistaken
    for the Q nadir.
    """
    half = _ms_to_samples(search_window_ms, sampling_rate)
    inner = _ms_to_samples(_QS_INNER_MARGIN_MS, sampling_rate)
    start = max(0, int(r_idx) - half)
    end = max(start + 2, int(r_idx) - inner)
    if end - start < 2:
        return None
    seg = ecg[start:end]
    rel = _segment_argext(seg, inverted=inverted)
    return start + rel


def find_s_wave_after_r(
    ecg: np.ndarray,
    r_idx: int,
    sampling_rate: float,
    *,
    search_window_ms: float,
    inverted: bool = False,
) -> Optional[int]:
    """
    Deepest S trough after R within ``[R + inner_margin, R + window]``.

    If the initial minimum is still on the descending QRS limb, retry on the
    late window ``[R + 40 ms, R + window]`` so S is not anchored on the downstroke.
    """
    half = _ms_to_samples(search_window_ms, sampling_rate)
    inner = _ms_to_samples(_QS_INNER_MARGIN_MS, sampling_rate)
    start = min(len(ecg) - 2, int(r_idx) + inner)
    end = min(len(ecg), int(r_idx) + half + 1)
    if end - start < 2:
        return None

    seg = ecg[start:end]
    rel = _segment_argext(seg, inverted=inverted)
    s_idx = start + rel

    if not inverted and _is_still_descending(ecg, s_idx, sampling_rate):
        retry_lo = int(r_idx) + _ms_to_samples(_S_DOWNSTROKE_RETRY_LO_MS, sampling_rate)
        retry_start = max(start, retry_lo)
        if end - retry_start >= 2:
            retry_seg = ecg[retry_start:end]
            retry_rel = _segment_argext(retry_seg, inverted=inverted)
            s_idx = retry_start + retry_rel

    return int(s_idx)


def find_q_s_waves_near_r(
    ecg: np.ndarray,
    r_idx: int,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    inverted: bool = False,
) -> Tuple[Optional[int], Optional[int]]:
    """Q and S sample indices for one R peak within the configured Q/S window."""
    q_idx = find_q_wave_before_r(
        ecg,
        r_idx,
        sampling_rate,
        search_window_ms=q_search_before_r_ms(cfg),
        inverted=inverted,
    )
    s_idx = find_s_wave_after_r(
        ecg,
        r_idx,
        sampling_rate,
        search_window_ms=s_search_after_r_ms(cfg),
        inverted=inverted,
    )
    return q_idx, s_idx
