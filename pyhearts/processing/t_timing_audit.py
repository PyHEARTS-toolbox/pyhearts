"""Optional per-cycle audit columns for record-level T timing diagnosis (Step A)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

AUDIT_KEYS = (
    "T_pre_record_center_idx",
    "T_record_t_guess_center_idx",
    "T_record_refined_center_idx",
    "record_t_s_anchor_idx",
    "record_t_q_anchor_idx",
    "template_rt_offset_ms",
    "t_refine_delta_samples",
)


def init_t_timing_audit(output_dict: Dict, n_cycles: int) -> None:
    n = int(n_cycles)
    for key in AUDIT_KEYS:
        if key not in output_dict:
            output_dict[key] = [np.nan] * n


def set_t_timing_audit(
    output_dict: Dict,
    cycle_idx: int,
    *,
    t_pre: Optional[float] = None,
    t_record_guess: Optional[float] = None,
    t_refined: Optional[float] = None,
    s_anchor: Optional[float] = None,
    q_anchor: Optional[float] = None,
    template_rt_ms: Optional[float] = None,
    refine_delta_samples: Optional[float] = None,
) -> None:
    if "T_pre_record_center_idx" not in output_dict:
        return
    mapping = {
        "T_pre_record_center_idx": t_pre,
        "T_record_t_guess_center_idx": t_record_guess,
        "T_record_refined_center_idx": t_refined,
        "record_t_s_anchor_idx": s_anchor,
        "record_t_q_anchor_idx": q_anchor,
        "template_rt_offset_ms": template_rt_ms,
        "t_refine_delta_samples": refine_delta_samples,
    }
    for key, val in mapping.items():
        if val is not None and cycle_idx < len(output_dict[key]):
            output_dict[key][cycle_idx] = val
