"""
Record-template biphasic +- morphology: classify S→Q template; optional guardrail / lobe search.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from pyhearts.config import ProcessCycleConfig

MORPH_BIPHASIC_POS_NEG = "biphasic_positive_negative"
LANDMARK_BIPHASIC_POSITIVE_APEX = "biphasic_positive_apex"


def biphasic_pm_classification_enabled(cfg: ProcessCycleConfig) -> bool:
    return bool(
        getattr(cfg, "record_template_detect_biphasic_positive_negative", False)
        or getattr(cfg, "record_biphasic_pm_early_t_guardrail", False)
        or getattr(cfg, "record_biphasic_pm_lobe_search", False)
    )


def biphasic_pm_lobe_search_enabled(cfg: ProcessCycleConfig) -> bool:
    return bool(getattr(cfg, "record_biphasic_pm_lobe_search", False))


def _morphology_segment_indices(n: int, cfg: ProcessCycleConfig) -> Tuple[int, int]:
    morph_frac = getattr(cfg, "record_template_t_morphology_sq_frac", None)
    if morph_frac is not None and n >= 2:
        lo_f, hi_f = morph_frac
        i0 = int(round(float(lo_f) * (n - 1)))
        i1 = int(round(float(hi_f) * (n - 1)))
        i0 = max(0, min(i0, n - 2))
        i1 = max(i0 + 1, min(i1, n - 1))
        return i0, i1
    mid = n // 2
    return 0, max(0, min(mid, n - 1))


def _st_baseline(template: np.ndarray, i0: int) -> float:
    """ST baseline from early template samples before morphology window."""
    ref_end = max(1, min(i0, template.size - 1))
    return float(np.median(template[: ref_end + 1]))


def _find_pm_candidates(
    seg: np.ndarray,
    baseline: float,
    fs: float,
) -> Tuple[List[dict], List[dict]]:
    rel = seg.astype(float) - baseline
    if rel.size < 3:
        return [], []
    prom_thresh = max(0.01, 0.12 * float(np.std(rel)))
    dist = max(1, int(round(0.03 * fs)))
    pos_idx, _ = find_peaks(rel, prominence=prom_thresh, distance=dist)
    neg_idx, _ = find_peaks(-rel, prominence=prom_thresh, distance=dist)
    pos: List[dict] = []
    neg: List[dict] = []
    for i in pos_idx:
        prom = float(peak_prominences(rel, [i])[0][0])
        pos.append({"idx": int(i), "signed_amp": float(rel[i]), "prominence": prom})
    for i in neg_idx:
        prom = float(peak_prominences(-rel, [i])[0][0])
        neg.append({"idx": int(i), "signed_amp": float(rel[i]), "prominence": prom})
    return pos, neg


def classify_biphasic_positive_negative(
    template: np.ndarray,
    cfg: ProcessCycleConfig,
    sampling_rate: float,
) -> Tuple[str, Optional[int], Optional[int]]:
    """
  Classify template ST–T morphology on the S→Q median beat.

  Returns (morphology_tag, positive_landmark_tpl_idx, negative_landmark_tpl_idx).
  Only ``biphasic_positive_negative`` triggers specialized handling; other tags
  mean no change to legacy pipeline.
    """
    if not biphasic_pm_classification_enabled(cfg):
        return "unchanged", None, None

    tmpl = np.asarray(template, dtype=float)
    n = tmpl.size
    if n < 8:
        return "unchanged", None, None

    i0, i1 = _morphology_segment_indices(n, cfg)
    seg = tmpl[i0 : i1 + 1]
    baseline = _st_baseline(tmpl, i0)
    pos, neg = _find_pm_candidates(seg, baseline, sampling_rate)

    if not pos or not neg:
        return "unchanged", None, None

    dom_pos = max(pos, key=lambda c: abs(c["signed_amp"]))
    dom_neg = max(neg, key=lambda c: abs(c["signed_amp"]))
    pos_tpl = i0 + int(dom_pos["idx"])
    neg_tpl = i0 + int(dom_neg["idx"])

    if dom_pos["idx"] < dom_neg["idx"]:
        return MORPH_BIPHASIC_POS_NEG, pos_tpl, neg_tpl
    return "unchanged", None, None


def apply_biphasic_positive_negative_landmark(
    t_j: int,
    t_landmark_source: str,
    pos_tpl: int,
    neg_tpl: int,
) -> Tuple[int, str, str, int, int]:
    """Force Tⱼ to positive apex; do not use rising_edge / isoelectric upslope."""
    return (
        int(pos_tpl),
        LANDMARK_BIPHASIC_POSITIVE_APEX,
        MORPH_BIPHASIC_POS_NEG,
        int(pos_tpl),
        int(neg_tpl),
    )


def _finite(val) -> bool:
    return val is not None and not (isinstance(val, float) and np.isnan(val))


def apply_biphasic_pm_early_t_guardrail(
    output_dict: Dict,
    epochs_df: pd.DataFrame,
    cycle_labels: np.ndarray,
    r_peaks: np.ndarray,
    ecg_delim: np.ndarray,
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Dict[str, int]:
    """
    Classify-only guardrail: prevent T from sitting before the template positive apex.

    For confident biphasic +− only, if ``T < first_positive_peak - margin_ms``, set
    ``T = first_positive_peak``. Never moves T later.
    """
    from pyhearts.processing.fiducial_provenance import set_wave_source
    from pyhearts.processing.record_delineation import (
        _stpq_s_q_anchor_indices,
        _sync_peak,
    )
    from pyhearts.processing.record_fiducial_smoothing import _sync_cycle_relative_peak
    from pyhearts.processing.record_stpq_detection import _tpl_index_to_sample

    stats: Dict[str, int] = {
        "adjusted": 0,
        "skipped": 0,
        "skipped_order": 0,
        "skipped_prominence": 0,
        "skipped_uplift_cap": 0,
    }
    if not getattr(cfg, "record_biphasic_pm_early_t_guardrail", False):
        stats["skipped"] = 1
        return stats
    if (
        tmpl is None
        or not getattr(tmpl, "valid", False)
        or str(getattr(tmpl, "t_morphology", "") or "") != MORPH_BIPHASIC_POS_NEG
    ):
        stats["skipped"] = 1
        return stats

    pos_tpl = getattr(tmpl, "t_biphasic_pos_landmark_idx", None)
    neg_tpl = getattr(tmpl, "t_biphasic_neg_landmark_idx", None)
    n_tpl = int(tmpl.template.size) if getattr(tmpl, "template", None) is not None else 0
    if pos_tpl is None or n_tpl < 2:
        stats["skipped"] = 1
        return stats

    beat_order = bool(
        getattr(cfg, "record_biphasic_pm_early_guardrail_beat_order_check", False)
    )
    min_prom = float(
        getattr(cfg, "record_biphasic_pm_early_guardrail_min_prominence_frac", 0.12)
    )

    margin = max(
        0.0, float(getattr(cfg, "record_biphasic_pm_early_guardrail_margin_ms", 10.0))
    )
    margin_samp = max(0, int(round(margin * float(sampling_rate) / 1000.0)))
    fs = float(sampling_rate)
    t_global = output_dict.get("T_global_center_idx", [])

    for cycle_idx, cycle_label in enumerate(cycle_labels):
        if cycle_idx >= len(t_global):
            break
        t_val = t_global[cycle_idx] if cycle_idx < len(t_global) else np.nan
        if not _finite(t_val):
            continue
        epoch_i = int(cycle_label)
        if epoch_i < 0 or epoch_i >= len(r_peaks):
            continue
        r_det = int(r_peaks[epoch_i])
        r_next = int(r_peaks[epoch_i + 1]) if epoch_i + 1 < len(r_peaks) else None
        s_i, q_next = _stpq_s_q_anchor_indices(
            ecg_delim, r_det, r_next, fs, cfg
        )
        if s_i is None or q_next is None:
            continue
        first_pos = _tpl_index_to_sample(int(s_i), int(q_next), float(pos_tpl), n_tpl)
        if first_pos is None:
            continue

        if beat_order:
            if neg_tpl is None:
                stats["skipped_order"] += 1
                continue
            t_neg = _tpl_index_to_sample(
                int(s_i), int(q_next), float(neg_tpl), n_tpl
            )
            min_sep = max(1, int(round(0.015 * fs)))
            if t_neg is None or int(first_pos) >= int(t_neg) - min_sep:
                stats["skipped_order"] += 1
                continue

        if min_prom > 0.0:
            st_lo = max(int(s_i), int(first_pos) - int(round(0.08 * fs)))
            st_hi = min(int(q_next), int(first_pos) + max(1, int(round(0.02 * fs))))
            seg = ecg_delim[st_lo : st_hi + 1].astype(float, copy=False)
            if seg.size >= 3:
                rel_i = int(first_pos) - st_lo
                rel_i = max(0, min(rel_i, seg.size - 1))
                baseline = float(np.median(seg[: max(1, rel_i)]))
                rel = seg - baseline
                peak_prom = float(rel[rel_i])
                peak_max = float(np.max(rel))
                if peak_prom <= 0.0 or (
                    peak_max > 0.0 and peak_prom < min_prom * peak_max
                ):
                    stats["skipped_prominence"] += 1
                    continue

        floor_idx = int(first_pos) - margin_samp
        if float(t_val) < float(floor_idx):
            uplift = float(first_pos) - float(t_val)
            max_up = max(
                0.0,
                float(
                    getattr(cfg, "record_biphasic_pm_early_guardrail_max_uplift_ms", 80.0)
                ),
            )
            max_up_samp = int(round(max_up * fs / 1000.0))
            if uplift <= 0 or uplift > max_up_samp:
                stats["skipped_uplift_cap"] += 1
                continue
            t_new = float(first_pos)
            t_global[cycle_idx] = t_new
            one_cycle = epochs_df.loc[epochs_df["cycle"] == cycle_label].sort_values(
                "index"
            )
            if not one_cycle.empty:
                sig = one_cycle["signal_y"].values.astype(float)
                t_ref = _sync_peak(
                    output_dict, cycle_idx, "T", t_new, one_cycle, fs, cfg
                )
                if _finite(t_ref):
                    _sync_cycle_relative_peak(
                        output_dict, cycle_idx, "T", t_ref, one_cycle, sig, fs, cfg
                    )
            set_wave_source(
                output_dict,
                cycle_idx,
                "T",
                "biphasic_pm_early_guardrail",
                confidence="medium",
            )
            stats["adjusted"] += 1

    output_dict["T_global_center_idx"] = list(t_global)
    return stats
