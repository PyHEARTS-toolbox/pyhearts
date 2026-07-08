"""
Recompute shape, interval, ST, and QTc features after fiducial timing updates.

Used by record-level delineation (Tier B1) when P/T/R center indices change.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from pyhearts.config import ProcessCycleConfig
from pyhearts.feature import calc_intervals, extract_shape_features
from pyhearts.feature.intervals import calc_qtc_all_formulas, interval_ms
from pyhearts.feature.st_segment import extract_st_segment_features
from pyhearts.processing.gaussian import compute_gauss_std
from pyhearts.processing.peaks import cycle_rel_to_global_sample, global_index_to_cycle_relative
from pyhearts.processing.qrs_boundary_detection_v2 import (
    detect_qrs_end_derivative,
    resolve_qrs_onset_idx,
)

_SHAPE_FEATURE_KEYS = [
    "duration_ms",
    "ri_idx",
    "le_idx",
    "rise_ms",
    "decay_ms",
    "rdsm",
    "sharpness",
    "max_upslope_mv_per_s",
    "max_downslope_mv_per_s",
    "slope_asymmetry",
]

_INTERVAL_PEAK_KEYS = [
    "P_le_idx",
    "P_ri_idx",
    "Q_le_idx",
    "Q_ri_idx",
    "R_le_idx",
    "R_ri_idx",
    "S_le_idx",
    "S_ri_idx",
    "T_le_idx",
    "T_ri_idx",
]

_WAVES = ("P", "Q", "R", "S", "T")


def _finite(val) -> bool:
    return val is not None and not (isinstance(val, float) and np.isnan(val))


def _get_cycle_series(output_dict: Dict, key: str, cycle_idx: int):
    arr = output_dict.get(key, [])
    if cycle_idx >= len(arr):
        return np.nan
    return arr[cycle_idx]


def _build_peak_data_for_cycle(
    output_dict: Dict,
    cycle_idx: int,
    xs_samples: np.ndarray,
    sig: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Dict:
    """Reconstruct peak_data from output_dict timing fields for one cycle."""
    peak_data: Dict = {}

    guesses = {}
    for comp in _WAVES:
        center = _get_cycle_series(output_dict, f"{comp}_center_idx", cycle_idx)
        height = _get_cycle_series(output_dict, f"{comp}_center_voltage", cycle_idx)
        if not _finite(center):
            continue
        center_f = float(center)
        height_f = float(height) if _finite(height) else float(sig[int(np.clip(round(center_f), 0, len(sig) - 1))])
        guesses[comp] = (int(round(center_f)), height_f)

    std_dict = compute_gauss_std(sig, guesses) if guesses else {}

    for comp in _WAVES:
        center = _get_cycle_series(output_dict, f"{comp}_center_idx", cycle_idx)
        if not _finite(center):
            if comp in ("Q", "S"):
                peak_data[comp] = {}
            continue

        center_f = float(center)
        height = _get_cycle_series(output_dict, f"{comp}_center_voltage", cycle_idx)
        if not _finite(height):
            height = float(sig[int(np.clip(round(center_f), 0, len(sig) - 1))])

        stdev = std_dict.get(comp)
        if stdev is None and comp in guesses:
            stdev = max(2.0, abs(guesses[comp][1]) * 0.1)

        entry = {
            "center_idx": center_f,
            "gauss_center": center_f,
            "gauss_height": float(height),
            "gauss_stdev_samples": float(stdev) if stdev is not None else np.nan,
            "gauss_fwhm_samples": (
                float(stdev) * 2.3548 if stdev is not None else np.nan
            ),
            "gauss_stdev_ms": (
                (float(stdev) / sampling_rate) * 1000.0 if stdev is not None else np.nan
            ),
            "gauss_fwhm_ms": (
                (float(stdev) * 2.3548 / sampling_rate) * 1000.0
                if stdev is not None
                else np.nan
            ),
            "le_idx": _get_cycle_series(output_dict, f"{comp}_le_idx", cycle_idx),
            "ri_idx": _get_cycle_series(output_dict, f"{comp}_ri_idx", cycle_idx),
        }
        peak_data[comp] = entry

    if "Q" not in peak_data:
        peak_data["Q"] = {}
    if "S" not in peak_data:
        peak_data["S"] = {}

    return peak_data


def _refresh_qrs_boundaries(
    peak_data: Dict,
    sig: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    cycle_idx: int,
    verbose: bool,
) -> None:
    r_center = peak_data.get("R", {}).get("center_idx")
    if r_center is None or not _finite(r_center):
        return
    r_center = int(round(float(r_center)))

    q_peak = peak_data.get("Q", {}).get("center_idx")
    q_peak_i = int(round(float(q_peak))) if q_peak is not None and _finite(q_peak) else None

    try:
        onset = resolve_qrs_onset_idx(
            sig,
            r_center,
            sampling_rate,
            q_peak_idx=q_peak_i,
            search_window_ms=cfg.qrs_onset_search_window_ms,
            fallback_offset_ms=cfg.qrs_onset_fallback_offset_ms,
            min_before_r_ms=cfg.qrs_onset_min_before_r_ms,
            max_before_r_ms=cfg.qrs_onset_max_before_r_ms,
            verbose=False,
            cycle_idx=cycle_idx,
        )
        peak_data.setdefault("Q", {})["le_idx"] = float(onset)
    except Exception:
        pass

    s_center = peak_data.get("S", {}).get("center_idx")
    s_center_i = int(round(float(s_center))) if s_center is not None and _finite(s_center) else None
    try:
        qrs_end = detect_qrs_end_derivative(
            sig,
            s_peak_idx=s_center_i,
            r_peak_idx=r_center,
            sampling_rate=sampling_rate,
            search_window_ms=100.0,
            verbose=False,
            cycle_idx=cycle_idx,
        )
        peak_data.setdefault("S", {})["ri_idx"] = float(qrs_end)
    except Exception:
        if r_center is not None:
            peak_data.setdefault("S", {})["ri_idx"] = float(
                min(len(sig) - 1, r_center + int(round(40 * sampling_rate / 1000.0)))
            )


def _assign_shape_and_peaks_to_output(
    output_dict: Dict,
    cycle_idx: int,
    peak_data: Dict,
    shape: Dict,
    xs_samples: np.ndarray,
    sig: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> None:
    valid = shape.get("valid_components", [])
    for comp in valid:
        comp_vals = shape.get("per_component", {}).get(comp, {})
        for key in _SHAPE_FEATURE_KEYS:
            dict_key = f"{comp}_{key}"
            if dict_key not in output_dict:
                continue
            value = float(comp_vals.get(key, np.nan))
            output_dict[dict_key][cycle_idx] = None if np.isnan(value) else round(value, 5)
        vint_key = f"{comp}_voltage_integral_uv_ms"
        if vint_key in output_dict:
            v = comp_vals.get("voltage_integral_uv_ms", np.nan)
            output_dict[vint_key][cycle_idx] = None if np.isnan(v) else round(float(v), 5)

        if comp in peak_data:
            for key in _SHAPE_FEATURE_KEYS:
                if key in comp_vals:
                    peak_data[comp][key] = comp_vals[key]
            peak_data[comp]["voltage_integral_uv_ms"] = comp_vals.get(
                "voltage_integral_uv_ms", np.nan
            )

    diffs = shape.get("pairwise_differences") or shape.get("global_metrics", {}).get(
        "interdeflection_voltage_differences", {}
    )
    for diff_name, diff_value in diffs.items():
        if diff_name not in output_dict:
            continue
        if diff_value is None or (isinstance(diff_value, float) and np.isnan(diff_value)):
            output_dict[diff_name][cycle_idx] = None
        else:
            output_dict[diff_name][cycle_idx] = round(float(diff_value), 5)

    for comp, pdata in peak_data.items():
        if not pdata:
            continue
        center_idx = pdata.get("center_idx", np.nan)
        le_idx = pdata.get("le_idx", np.nan)
        ri_idx = pdata.get("ri_idx", np.nan)

        if _finite(center_idx):
            gci = cycle_rel_to_global_sample(
                float(center_idx),
                xs_samples,
                sig,
                refine_subsample=(
                    cfg.use_subsample_peak_refinement and comp in ("P", "R", "T")
                ),
            )
            output_dict[f"{comp}_global_center_idx"][cycle_idx] = gci
            output_dict[f"{comp}_center_idx"][cycle_idx] = float(center_idx)
            i_n = int(np.clip(round(float(center_idx)), 0, len(sig) - 1))
            output_dict[f"{comp}_center_voltage"][cycle_idx] = float(sig[i_n])
            output_dict[f"{comp}_center_ms"][cycle_idx] = (
                float(center_idx) / sampling_rate
            ) * 1000.0

        for edge, key_suffix in ((le_idx, "le"), (ri_idx, "ri")):
            if _finite(edge):
                edge_i = int(np.clip(round(float(edge)), 0, len(xs_samples) - 1))
                for out_key, val in (
                    (f"{comp}_{key_suffix}_idx", float(edge)),
                    (f"{comp}_global_{key_suffix}_idx", float(xs_samples[edge_i])),
                    (
                        f"{comp}_{key_suffix}_ms",
                        (float(edge) / sampling_rate) * 1000.0,
                    ),
                ):
                    if out_key in output_dict:
                        output_dict[out_key][cycle_idx] = val
                volt_key = f"{comp}_{key_suffix}_voltage"
                if volt_key in output_dict and 0 <= edge_i < len(sig):
                    output_dict[volt_key][cycle_idx] = float(sig[edge_i])

        gauss_center = pdata.get("gauss_center", np.nan)
        gauss_fwhm = pdata.get("gauss_fwhm_samples", np.nan)
        if _finite(gauss_center) and _finite(gauss_fwhm):
            half = float(gauss_fwhm) / 2.0
            left = max(0, min(int(round(gauss_center - half)), len(xs_samples) - 1))
            right = max(0, min(int(round(gauss_center + half)), len(xs_samples) - 1))
            if right >= left:
                fwhm_pairs = (
                    ("fwhm_le_idx", float(left)),
                    ("fwhm_ri_idx", float(right)),
                    ("fwhm_le_ms", (left / sampling_rate) * 1000.0),
                    ("fwhm_ri_ms", (right / sampling_rate) * 1000.0),
                    ("fwhm_global_le_idx", float(xs_samples[left])),
                    ("fwhm_global_ri_idx", float(xs_samples[right])),
                )
                for suffix, val in fwhm_pairs:
                    out_key = f"{comp}_{suffix}"
                    if out_key in output_dict:
                        output_dict[out_key][cycle_idx] = val

        for gkey in (
            "gauss_center",
            "gauss_height",
            "gauss_stdev_samples",
            "gauss_fwhm_samples",
            "gauss_stdev_ms",
            "gauss_fwhm_ms",
        ):
            out_key = f"{comp}_{gkey}"
            if out_key in output_dict and gkey in pdata:
                val = pdata[gkey]
                output_dict[out_key][cycle_idx] = (
                    None if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
                )


def _refresh_intervals_and_st_for_cycle(
    output_dict: Dict,
    cycle_idx: int,
    peak_data: Dict,
    sig: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    previous_r_global: Optional[float],
    previous_p_global: Optional[float],
) -> None:
    peak_series = {key: output_dict.get(key, []) for key in _INTERVAL_PEAK_KEYS}

    run_intervals = (
        not cfg.lite_mode or cfg.record_delineation_refresh_shape
    )
    if run_intervals:
        interval_results = calc_intervals(
            all_peak_series=peak_series,
            cycle_idx=cycle_idx,
            sampling_rate=int(sampling_rate),
            window_size=3,
        )
        for interval_name, value in interval_results.items():
            if interval_name in output_dict:
                output_dict[interval_name][cycle_idx] = value
    else:
        interval_results = {}

    if not cfg.lite_mode:
        try:
            s_ri = peak_series["S_ri_idx"][cycle_idx] if cycle_idx < len(peak_series["S_ri_idx"]) else np.nan
            t_le = peak_series["T_le_idx"][cycle_idx] if cycle_idx < len(peak_series["T_le_idx"]) else np.nan
            p_ri = peak_series["P_ri_idx"][cycle_idx] if cycle_idx < len(peak_series["P_ri_idx"]) else np.nan
            q_le = peak_series["Q_le_idx"][cycle_idx] if cycle_idx < len(peak_series["Q_le_idx"]) else np.nan

            st_features = extract_st_segment_features(
                signal=sig,
                s_ri_idx=int(s_ri) if _finite(s_ri) else None,
                t_le_idx=int(t_le) if _finite(t_le) else None,
                p_ri_idx=int(p_ri) if _finite(p_ri) else None,
                q_le_idx=int(q_le) if _finite(q_le) else None,
                sampling_rate=sampling_rate,
                j_point_offset_ms=60.0,
                verbose=False,
            )
            for feature_name, value in st_features.items():
                if feature_name in output_dict:
                    output_dict[feature_name][cycle_idx] = value
        except Exception:
            pass

    r_g = _get_cycle_series(output_dict, "R_global_center_idx", cycle_idx)
    p_g = _get_cycle_series(output_dict, "P_global_center_idx", cycle_idx)

    ms_per_sample = 1000.0 / sampling_rate
    lo_rr_ms, hi_rr_ms = cfg.rr_bounds_ms
    lo_pp_ms, hi_pp_ms = cfg.pp_bounds_ms or cfg.rr_bounds_ms

    if "RR_interval_ms" in output_dict:
        output_dict["RR_interval_ms"][cycle_idx] = interval_ms(
            r_g if _finite(r_g) else None,
            previous_r_global,
            lo_rr_ms,
            hi_rr_ms,
            ms_per_sample,
        )
    if "PP_interval_ms" in output_dict:
        output_dict["PP_interval_ms"][cycle_idx] = interval_ms(
            p_g if _finite(p_g) else None,
            previous_p_global,
            lo_pp_ms,
            hi_pp_ms,
            ms_per_sample,
        )

    if not cfg.lite_mode:
        qt_ms = _get_cycle_series(output_dict, "QT_interval_ms", cycle_idx)
        rr_ms = _get_cycle_series(output_dict, "RR_interval_ms", cycle_idx)
        if _finite(qt_ms) and _finite(rr_ms) and float(rr_ms) > 0:
            for qtc_name, qtc_value in calc_qtc_all_formulas(float(qt_ms), float(rr_ms)).items():
                if qtc_name in output_dict:
                    output_dict[qtc_name][cycle_idx] = qtc_value


def refresh_cycles_after_timing_update(
    output_dict: Dict,
    epochs_df: pd.DataFrame,
    cycle_labels: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    modified_cycles: Iterable[int],
    *,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Re-run shape extraction and interval metrics for cycles with updated timing.
    """
    stats = {"cycles_refreshed": 0, "shape_ok": 0, "skipped": 0}
    if not cfg.record_delineation_refresh_features:
        stats["skipped"] = 1
        return stats

    modified = sorted({int(c) for c in modified_cycles})
    if not modified:
        return stats

    for cycle_idx in modified:
        if cycle_idx >= len(cycle_labels):
            continue
        cycle_label = cycle_labels[cycle_idx]
        one_cycle = epochs_df.loc[epochs_df["cycle"] == cycle_label].sort_values("index")
        if one_cycle.empty:
            continue

        if "index" in one_cycle.columns:
            xs_samples = one_cycle["index"].values.astype(int)
        else:
            xs_samples = one_cycle["signal_x"].values.astype(int)
        sig = one_cycle["signal_y"].values.astype(float)
        if len(sig) < 3:
            continue

        peak_data = _build_peak_data_for_cycle(
            output_dict, cycle_idx, xs_samples, sig, sampling_rate, cfg
        )
        _refresh_qrs_boundaries(
            peak_data, sig, sampling_rate, cfg, cycle_idx=cycle_idx, verbose=verbose
        )

        shape = {
            "valid_components": [],
            "per_component": {},
            "pairwise_differences": {},
        }
        if cfg.record_delineation_refresh_shape:
            component_labels = [c for c in peak_data if peak_data[c].get("center_idx") is not None]
            if component_labels and "R" in peak_data:
                gauss_center = np.array(
                    [float(peak_data[c]["gauss_center"]) for c in component_labels]
                )
                gauss_height = np.array(
                    [float(peak_data[c]["gauss_height"]) for c in component_labels]
                )
                gauss_stdev = np.array(
                    [
                        float(peak_data[c]["gauss_stdev_samples"])
                        if _finite(peak_data[c].get("gauss_stdev_samples"))
                        else 2.5
                        for c in component_labels
                    ]
                )
                r_height = float(peak_data["R"]["gauss_height"])
                precomputed_bounds = {}
                if (
                    "Q" in peak_data
                    and _finite(peak_data["Q"].get("le_idx"))
                    and _finite(peak_data["Q"].get("center_idx"))
                ):
                    precomputed_bounds["Q"] = (
                        int(peak_data["Q"]["le_idx"]),
                        int(peak_data["Q"]["center_idx"]),
                    )
                if (
                    "S" in peak_data
                    and _finite(peak_data["S"].get("ri_idx"))
                    and _finite(peak_data["S"].get("center_idx"))
                ):
                    precomputed_bounds["S"] = (
                        int(peak_data["S"]["center_idx"]),
                        int(peak_data["S"]["ri_idx"]),
                    )
                fit = np.zeros_like(sig)
                try:
                    shape = extract_shape_features(
                        signal=fit,
                        gauss_centers=gauss_center,
                        gauss_stdevs=gauss_stdev,
                        gauss_heights=gauss_height,
                        component_labels=component_labels,
                        r_height=r_height,
                        sampling_rate=int(sampling_rate),
                        cfg=cfg,
                        verbose=False,
                        precomputed_bounds=precomputed_bounds or None,
                    )
                    stats["shape_ok"] += 1
                except Exception as exc:
                    if verbose:
                        print(f"[refresh] cycle {cycle_idx} shape failed: {exc}")

        _assign_shape_and_peaks_to_output(
            output_dict,
            cycle_idx,
            peak_data,
            shape,
            xs_samples,
            sig,
            sampling_rate,
            cfg,
        )

        prev_r = (
            float(output_dict["R_global_center_idx"][cycle_idx - 1])
            if cycle_idx > 0
            and cycle_idx - 1 < len(output_dict.get("R_global_center_idx", []))
            and _finite(output_dict["R_global_center_idx"][cycle_idx - 1])
            else None
        )
        prev_p = (
            float(output_dict["P_global_center_idx"][cycle_idx - 1])
            if cycle_idx > 0
            and cycle_idx - 1 < len(output_dict.get("P_global_center_idx", []))
            and _finite(output_dict["P_global_center_idx"][cycle_idx - 1])
            else None
        )
        _refresh_intervals_and_st_for_cycle(
            output_dict,
            cycle_idx,
            peak_data,
            sig,
            sampling_rate,
            cfg,
            previous_r_global=prev_r,
            previous_p_global=prev_p,
        )
        stats["cycles_refreshed"] += 1

    return stats
