"""
Per-cycle provenance for P/T timing (source and confidence) and WFDB lead metadata.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np

PROVENANCE_KEYS = (
    "p_source",
    "t_source",
    "p_confidence",
    "t_confidence",
    "wfdb_lead_index",
    "wfdb_lead_name",
    "lead_policy",
)


def _is_finite(val) -> bool:
    return val is not None and not (isinstance(val, float) and np.isnan(val))


def init_fiducial_provenance(output_dict: Dict, n_cycles: int) -> None:
    """Add provenance lists to ``output_dict`` (idempotent for missing keys only)."""
    n = int(n_cycles)
    defaults: Dict[str, List] = {
        "p_source": [None] * n,
        "t_source": [None] * n,
        "p_confidence": [None] * n,
        "t_confidence": [None] * n,
        "wfdb_lead_index": [np.nan] * n,
        "wfdb_lead_name": [None] * n,
        "lead_policy": [None] * n,
    }
    for key, val in defaults.items():
        if key not in output_dict:
            output_dict[key] = list(val)


def set_wave_source(
    output_dict: Dict,
    cycle_idx: int,
    wave: str,
    source: str,
    *,
    confidence: Optional[str] = None,
) -> None:
    """Set ``p_source`` / ``t_source`` (and optional confidence) for one cycle."""
    wave = wave.upper()
    if wave not in ("P", "T"):
        raise ValueError(f"wave must be 'P' or 'T', got {wave!r}")
    src_key = f"{wave.lower()}_source"
    conf_key = f"{wave.lower()}_confidence"
    if src_key not in output_dict:
        return
    output_dict[src_key][cycle_idx] = source
    if confidence is not None and conf_key in output_dict:
        output_dict[conf_key][cycle_idx] = confidence


def mark_detected_pt_sources(output_dict: Dict, cycle_idx: int) -> None:
    """Tag P/T as per-cycle when global center indices are finite."""
    for wave, gkey in (("P", "P_global_center_idx"), ("T", "T_global_center_idx")):
        vals = output_dict.get(gkey, [])
        if cycle_idx >= len(vals):
            continue
        if _is_finite(vals[cycle_idx]):
            src_key = f"{wave.lower()}_source"
            if src_key in output_dict and output_dict[src_key][cycle_idx] is None:
                set_wave_source(
                    output_dict,
                    cycle_idx,
                    wave,
                    "per_cycle",
                    confidence="high",
                )


def set_run_lead_metadata(
    output_dict: Dict,
    *,
    lead_index: int,
    lead_name: str,
    lead_policy: str,
) -> None:
    """Fill lead metadata columns for all cycles."""
    if "wfdb_lead_index" not in output_dict:
        return
    n = len(output_dict["wfdb_lead_index"])
    output_dict["wfdb_lead_index"] = [int(lead_index)] * n
    output_dict["wfdb_lead_name"] = [str(lead_name)] * n
    output_dict["lead_policy"] = [str(lead_policy)] * n


def attach_provenance_to_dataframe(df, metadata: Optional[dict] = None):
    """
    Add constant lead/policy columns to a features DataFrame when metadata is set.

    Returns the same DataFrame (mutated in place) for chaining.
    """
    import pandas as pd

    if metadata is None:
        return df
    n = len(df)
    if n == 0:
        return df
    if "lead_index" in metadata:
        df["wfdb_lead_index"] = int(metadata["lead_index"])
    if "lead_name" in metadata:
        df["wfdb_lead_name"] = str(metadata["lead_name"])
    if "lead_policy" in metadata:
        df["lead_policy"] = str(metadata["lead_policy"])
    if "manual_ann_ext" in metadata:
        df["manual_ann_ext"] = str(metadata["manual_ann_ext"])
    return df
