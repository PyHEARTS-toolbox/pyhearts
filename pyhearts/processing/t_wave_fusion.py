"""
Record-level T-wave prior, secondary amplitude detector, and fusion for missing T peaks.

Applied as a post-pass after all cycles are processed (see PyHEARTS.analyze_ecg).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.derivative_t_detection import (
    compute_filtered_derivative,
    compute_t_search_window,
    detect_t_wave_derivative_based,
)
from pyhearts.processing.qrs_removal import remove_qrs_sigmoid
from pyhearts.processing.peaks import cycle_rel_to_global_sample, sample_at_fractional_index
from pyhearts.processing.validation import validate_peaks


@dataclass(frozen=True)
class RecordTPrior:
    """Median R→T interval estimated from beats with detected T peaks."""

    median_rt_samples: float
    mad_rt_samples: float
    inverted: bool
    n_beats: int
    valid: bool


def _is_finite_idx(val) -> bool:
    return val is not None and not (isinstance(val, float) and np.isnan(val))


def estimate_record_rt_prior(
    r_global: np.ndarray,
    t_global: np.ndarray,
    t_heights: Optional[np.ndarray] = None,
    *,
    min_beats: int = 5,
    default_rt_ms: float = 280.0,
    sampling_rate: float = 250.0,
) -> RecordTPrior:
    """Estimate record-level R→T timing from cycles with both R and T detected."""
    rt: List[float] = []
    heights: List[float] = []
    for i, (r, t) in enumerate(zip(r_global, t_global)):
        if not (_is_finite_idx(r) and _is_finite_idx(t)):
            continue
        rt.append(float(t) - float(r))
        if t_heights is not None and i < len(t_heights) and _is_finite_idx(t_heights[i]):
            heights.append(float(t_heights[i]))

    if len(rt) < min_beats:
        default_samples = default_rt_ms * sampling_rate / 1000.0
        return RecordTPrior(
            median_rt_samples=default_samples,
            mad_rt_samples=max(10.0, 0.08 * default_samples),
            inverted=False,
            n_beats=len(rt),
            valid=False,
        )

    rt_arr = np.asarray(rt, dtype=float)
    med = float(np.median(rt_arr))
    mad = float(np.median(np.abs(rt_arr - med)))
    if mad < 3.0:
        mad = max(10.0, 0.08 * med)

    inverted = False
    if t_heights is not None and len(heights) >= min_beats:
        h = np.asarray([heights[i] for i in range(len(rt))], dtype=float)
        inverted = bool(np.median(h) < 0)

    return RecordTPrior(
        median_rt_samples=med,
        mad_rt_samples=mad,
        inverted=inverted,
        n_beats=len(rt),
        valid=True,
    )


def detect_t_amplitude_peak(
    signal: np.ndarray,
    search_start: int,
    search_end: int,
    *,
    inverted: bool = False,
) -> Tuple[Optional[int], Optional[float]]:
    """Secondary T detector: signal-domain argmax/argmin in the search window."""
    if search_end <= search_start + 2:
        return None, None
    seg = signal[search_start:search_end]
    if len(seg) < 3:
        return None, None
    if inverted:
        rel = int(np.argmin(seg))
    else:
        rel = int(np.argmax(seg))
    idx = search_start + rel
    return idx, float(signal[idx])


def _prior_window(
    r_idx: int,
    t_start: int,
    t_end: int,
    prior: RecordTPrior,
    fs: float,
    cfg: ProcessCycleConfig,
) -> Tuple[int, int]:
    center = int(round(r_idx + prior.median_rt_samples))
    half_ms = max(
        cfg.t_wave_prior_window_ms,
        cfg.t_wave_prior_min_window_ms,
        prior.mad_rt_samples * fs / 1000.0 * cfg.t_wave_prior_max_deviation_mad,
    )
    half = int(round(half_ms * fs / 1000.0))
    return max(t_start, center - half), min(t_end, center + half)


def detect_t_fused(
    signal: np.ndarray,
    r_idx: int,
    r_height: float,
    sampling_rate: float,
    prior: RecordTPrior,
    cfg: ProcessCycleConfig,
    *,
    qrs_end_idx: Optional[int] = None,
    s_center_idx: Optional[int] = None,
    primary: Optional[Tuple[int, float]] = None,
) -> Tuple[Optional[int], Optional[float], str]:
    """
    Fuse primary derivative T, secondary amplitude T, and record-level prior.

    Returns (peak_idx, amplitude, source_tag).
    """
    if primary is not None and primary[0] is not None:
        return primary[0], primary[1], "primary"

    n = len(signal)
    t_start, t_end = compute_t_search_window(
        r_idx,
        n,
        sampling_rate,
        qrs_end_idx=qrs_end_idx,
        s_center_idx=s_center_idx,
        start_offset_ms=cfg.t_wave_search_start_ms,
        qrs_end_margin_ms=cfg.t_wave_search_qrs_end_margin_ms,
        rr_frac=cfg.t_wave_search_rr_frac,
        max_offset_ms=cfg.t_wave_search_max_ms,
        end_margin_ms=cfg.t_wave_search_end_margin_ms,
        min_window_ms=cfg.t_wave_search_min_window_ms,
    )
    if t_end - t_start < 3:
        return None, None, "none"

    sig_for_t = signal
    if cfg.t_wave_use_qrs_removal:
        sig_for_t = remove_qrs_sigmoid(
            signal,
            r_idx,
            sampling_rate,
            s_peak_idx=s_center_idx,
            qrs_end_idx=qrs_end_idx,
            pre_rr_frac=cfg.t_wave_qrs_pre_rr_frac,
            post_qrs_ms=cfg.t_wave_qrs_post_ms,
        )

    derivative = compute_filtered_derivative(sig_for_t, sampling_rate)

    candidates: List[Tuple[int, float, str, float]] = []

    if cfg.t_wave_use_secondary_detector:
        amp_idx, amp_h = detect_t_amplitude_peak(
            sig_for_t, t_start, t_end, inverted=prior.inverted
        )
        if amp_idx is not None and amp_h is not None:
            candidates.append((amp_idx, amp_h, "amplitude", 0.0))

        deriv_result = detect_t_wave_derivative_based(
            signal=sig_for_t,
            derivative=derivative,
            search_start=t_start,
            search_end=t_end,
            s_end_idx=s_center_idx,
            sampling_rate=sampling_rate,
            verbose=False,
            r_peak_idx=r_idx,
            r_peak_value=r_height,
            region_expansion_ms=cfg.t_wave_region_expansion_ms,
            region_min_fraction=cfg.t_wave_region_min_fraction,
        )
        if deriv_result[0] is not None:
            candidates.append((deriv_result[0], deriv_result[3], "derivative", 0.0))

    if cfg.t_wave_use_record_prior and prior.valid:
        p0, p1 = _prior_window(r_idx, t_start, t_end, prior, sampling_rate, cfg)
        if p1 - p0 >= 3:
            prior_idx, prior_h = detect_t_amplitude_peak(
                sig_for_t, p0, p1, inverted=prior.inverted
            )
            if prior_idx is not None and prior_h is not None:
                pred = r_idx + prior.median_rt_samples
                dist = abs(prior_idx - pred)
                candidates.append((prior_idx, prior_h, "prior", dist))

            deriv_prior = detect_t_wave_derivative_based(
                signal=sig_for_t,
                derivative=derivative,
                search_start=p0,
                search_end=p1,
                s_end_idx=s_center_idx,
                sampling_rate=sampling_rate,
                verbose=False,
                r_peak_idx=r_idx,
                r_peak_value=r_height,
                region_expansion_ms=cfg.t_wave_region_expansion_ms,
                region_min_fraction=cfg.t_wave_region_min_fraction,
            )
            if deriv_prior[0] is not None:
                pred = r_idx + prior.median_rt_samples
                dist = abs(deriv_prior[0] - pred)
                candidates.append((deriv_prior[0], deriv_prior[3], "prior_deriv", dist))

    if not candidates:
        return None, None, "none"

    pred = r_idx + prior.median_rt_samples
    max_dev = prior.mad_rt_samples * cfg.t_wave_prior_max_deviation_mad
    if not prior.valid:
        max_dev = max(max_dev, 0.25 * (t_end - t_start))

    best: Optional[Tuple[int, float, str]] = None
    best_score = np.inf
    for idx, h, tag, dist in candidates:
        if prior.valid and dist > max_dev:
            continue
        score = dist if prior.valid else abs(idx - pred)
        if tag == "prior" and cfg.t_wave_fusion_prefer_prior_on_tie:
            score -= 0.5
        if score < best_score:
            best_score = score
            best = (idx, h, tag)

    if best is None:
        idx, h, tag, _ = min(candidates, key=lambda c: c[3])
        best = (idx, h, tag)

    return best[0], best[1], best[2]


def validate_t_candidate(
    t_idx: int,
    t_height: float,
    r_idx: int,
    r_height: float,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    prior_guided: bool = False,
) -> bool:
    """Validate a fused T candidate (relaxed amplitude when prior-guided)."""
    from dataclasses import replace as dc_replace

    cfg_use = cfg
    if prior_guided and cfg.t_wave_prior_amp_min_ratio is not None:
        ratios = dict(cfg.amp_min_ratio)
        ratios["T"] = cfg.t_wave_prior_amp_min_ratio
        cfg_use = dc_replace(cfg, amp_min_ratio=ratios)

    validated = validate_peaks(
        peaks={"T": (t_idx, t_height)},
        r_center_idx=r_idx,
        r_height=r_height,
        sampling_rate=sampling_rate,
        verbose=False,
        cycle_idx=None,
        cfg=cfg_use,
    )
    t_val = validated.get("T", (None, None))
    return t_val[0] is not None


def assign_t_peak_to_output(
    output_dict: Dict,
    cycle_idx: int,
    center_rel: int,
    height: float,
    xs_samples: np.ndarray,
    sig: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig | None = None,
) -> None:
    """Write T peak indices and basic voltages into output_dict."""
    if center_rel < 0 or center_rel >= len(xs_samples):
        return
    refine = cfg is not None and cfg.use_subsample_peak_refinement
    center_rel_f = float(center_rel)
    global_idx = cycle_rel_to_global_sample(
        center_rel_f,
        xs_samples,
        sig,
        refine_subsample=refine,
    )
    if refine:
        height = sample_at_fractional_index(sig, center_rel_f)
    output_dict["T_global_center_idx"][cycle_idx] = global_idx
    output_dict["T_center_idx"][cycle_idx] = center_rel_f
    i_nearest = int(np.clip(np.round(center_rel_f), 0, len(sig) - 1))
    output_dict["T_center_voltage"][cycle_idx] = float(sig[i_nearest])
    output_dict["T_gauss_height"][cycle_idx] = float(height)
    output_dict["T_center_ms"][cycle_idx] = (center_rel / sampling_rate) * 1000.0


def recover_missing_t_waves(
    output_dict: Dict,
    epochs_df: pd.DataFrame,
    cycle_labels: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Post-pass: fill missing T peaks using record prior + secondary detector.

    Returns stats dict with counts of recovery attempts and successes.
    """
    if not (cfg.t_wave_use_record_prior or cfg.t_wave_use_secondary_detector):
        return {"attempted": 0, "recovered": 0}

    r_global = np.asarray(output_dict.get("R_global_center_idx", []), dtype=float)
    t_global = np.asarray(output_dict.get("T_global_center_idx", []), dtype=float)
    t_voltage = np.asarray(output_dict.get("T_center_voltage", []), dtype=float)

    prior = estimate_record_rt_prior(
        r_global,
        t_global,
        t_voltage if len(t_voltage) == len(t_global) else None,
        min_beats=cfg.t_wave_prior_min_beats,
        default_rt_ms=cfg.t_wave_prior_default_rt_ms,
        sampling_rate=sampling_rate,
    )

    stats = {"attempted": 0, "recovered": 0, "prior_valid": int(prior.valid), "prior_n": prior.n_beats}

    for cycle_idx, cycle_label in enumerate(cycle_labels):
        if cycle_idx >= len(r_global):
            break
        if not _is_finite_idx(r_global[cycle_idx]):
            continue
        if _is_finite_idx(t_global[cycle_idx]):
            continue

        one_cycle = epochs_df.loc[epochs_df["cycle"] == cycle_label].sort_values("index")
        if one_cycle.empty:
            continue

        if "index" in one_cycle.columns:
            xs_samples = one_cycle["index"].values.astype(int)
        else:
            xs_samples = one_cycle["signal_x"].values.astype(int)
        sig = one_cycle["signal_y"].values.astype(float)

        r_g = int(r_global[cycle_idx])
        matches = np.where(xs_samples == r_g)[0]
        if len(matches) == 0:
            r_rel = int(np.argmin(np.abs(xs_samples - r_g)))
        else:
            r_rel = int(matches[0])

        r_height = output_dict.get("R_center_voltage", [np.nan] * len(r_global))
        r_h = (
            float(r_height[cycle_idx])
            if cycle_idx < len(r_height) and _is_finite_idx(r_height[cycle_idx])
            else float(sig[r_rel])
        )

        stats["attempted"] += 1

        t_idx, t_h, source = detect_t_fused(
            sig,
            r_rel,
            r_h,
            sampling_rate,
            prior,
            cfg,
            primary=None,
        )
        if t_idx is None or t_h is None:
            continue

        prior_guided = source in ("prior", "prior_deriv") or not prior.valid
        if not validate_t_candidate(
            t_idx, t_h, r_rel, r_h, sampling_rate, cfg, prior_guided=prior_guided
        ):
            continue

        assign_t_peak_to_output(
            output_dict, cycle_idx, t_idx, t_h, xs_samples, sig, sampling_rate, cfg=cfg
        )
        stats["recovered"] += 1
        if verbose:
            print(
                f"[T recovery] cycle {cycle_idx}: T at rel={t_idx} "
                f"(source={source}, prior_rt={prior.median_rt_samples:.1f} samples)"
            )

    return stats
