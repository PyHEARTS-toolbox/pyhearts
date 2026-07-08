"""
Per-beat P/T plausibility checks (RT interval, T apex dominance in search window).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from pyhearts.config import ProcessCycleConfig


def validate_p_pr_interval(
    p_center_idx: int,
    r_center_idx: int,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> bool:
    """
    Return True if P–R spacing is physiologically plausible.

    Rejects P peaks too close to R (QRS encroachment) or implausibly far before R.
    """
    if not cfg.p_pr_interval_validation:
        return True
    if p_center_idx is None or r_center_idx is None:
        return True
    pr_ms = (int(r_center_idx) - int(p_center_idx)) / float(sampling_rate) * 1000.0
    lo, hi = cfg.p_pr_interval_bounds_ms
    return lo <= pr_ms <= hi


def check_t_peak_dominance(
    signal: np.ndarray,
    peak_idx: int,
    search_start: int,
    search_end: int,
    morphology: int,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> bool:
    """
    Return True if the detected T apex is the dominant extremum in the search window.

    Rejects secondary deflections (U waves, repolarization tails) that sit in a
    plausible RT interval but are not the main T peak.
    """
    if not cfg.t_morphology_dominance_check:
        return True
    n = len(signal)
    if n < 3:
        return True
    lo = int(np.clip(search_start, 0, n - 1))
    hi = int(np.clip(search_end, lo + 1, n))
    seg = signal[lo:hi]
    if seg.size < 5:
        return True
    peak_idx = int(np.clip(peak_idx, lo, hi - 1))
    peak_rel = peak_idx - lo
    inverted = int(morphology) != 0
    if inverted:
        global_rel = int(np.argmin(seg))
        peak_val = float(seg[peak_rel])
        opp_val = float(np.max(seg))
    else:
        global_rel = int(np.argmax(seg))
        peak_val = float(seg[peak_rel])
        opp_val = float(np.min(seg))
    ptp = float(np.ptp(seg))
    if ptp <= 1e-9:
        return True
    prominence = abs(peak_val - opp_val)
    if prominence < cfg.t_dominance_min_fraction * ptp:
        return False
    tol = max(
        1,
        int(round(cfg.t_dominance_max_offset_ms * float(sampling_rate) / 1000.0)),
    )
    return abs(peak_rel - global_rel) <= tol


def _finite(val: Any) -> bool:
    return val is not None and not (isinstance(val, float) and np.isnan(val))


def _clear_t_at_cycle(output_dict: Dict, cycle_idx: int) -> None:
    for key in (
        "T_global_center_idx",
        "T_center_idx",
        "T_center_voltage",
        "T_center_ms",
    ):
        arr = output_dict.get(key)
        if arr is not None and cycle_idx < len(arr):
            arr[cycle_idx] = np.nan
    for key in ("t_source", "t_confidence"):
        arr = output_dict.get(key)
        if arr is not None and cycle_idx < len(arr):
            arr[cycle_idx] = None


def apply_rt_plausibility_gate(
    output_dict: Dict,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Null T fiducials that fail RT bounds or record-level delay outlier tests.

    Returns
    -------
    dict
        Counts: ``checked``, ``rejected_bounds``, ``rejected_outlier``, ``kept``.
    """
    stats = {
        "checked": 0,
        "rejected_bounds": 0,
        "rejected_outlier": 0,
        "kept": 0,
    }
    if not cfg.t_rt_plausibility_gate:
        return stats

    r_list = output_dict.get("R_global_center_idx", [])
    t_list = output_dict.get("T_global_center_idx", [])
    if not r_list or not t_list:
        return stats

    n = min(len(r_list), len(t_list))
    fs = float(sampling_rate)
    rt_min, rt_max = cfg.t_rt_bounds_ms

    delays: List[float] = []
    indices: List[int] = []
    for i in range(n):
        rv, tv = r_list[i], t_list[i]
        if not (_finite(rv) and _finite(tv)):
            continue
        rt_ms = (float(tv) - float(rv)) / fs * 1000.0
        if rt_min <= rt_ms <= rt_max:
            delays.append(rt_ms)
            indices.append(i)

    if not delays:
        return stats

    med_rt = float(np.median(delays))
    mad_rt = float(np.median(np.abs(np.asarray(delays) - med_rt)))
    if mad_rt < 1.0:
        mad_rt = max(8.0, 0.08 * abs(med_rt))
    fence = max(
        cfg.t_rt_plausibility_min_fence_ms,
        mad_rt * cfg.t_rt_plausibility_mad_multiplier,
    )

    for i in range(n):
        rv, tv = r_list[i], t_list[i]
        if not (_finite(rv) and _finite(tv)):
            continue
        stats["checked"] += 1
        rt_ms = (float(tv) - float(rv)) / fs * 1000.0
        if rt_ms < rt_min or rt_ms > rt_max:
            _clear_t_at_cycle(output_dict, i)
            stats["rejected_bounds"] += 1
            continue
        if abs(rt_ms - med_rt) > fence:
            _clear_t_at_cycle(output_dict, i)
            stats["rejected_outlier"] += 1
            continue
        stats["kept"] += 1

    if verbose and (stats["rejected_bounds"] or stats["rejected_outlier"]):
        print(
            f"[RT plausibility] median_RT={med_rt:.1f} ms, fence={fence:.1f} ms, "
            f"kept={stats['kept']}, bounds={stats['rejected_bounds']}, "
            f"outlier={stats['rejected_outlier']}"
        )
    return stats
