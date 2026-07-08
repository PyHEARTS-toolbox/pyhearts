"""
WFDB lead selection and manual annotation pairing for QTDB and similar databases.

Use the same policy for PyHEARTS analysis, benchmarks, exports, and Bland–Altman
comparisons so the analyzed trace matches expert ``q1c`` / ``q2c`` annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

LeadPolicy = Literal["first", "ecg2_else_ecg1", "limb_preferred", "second"]

LEAD_POLICIES: Tuple[str, ...] = (
    "first",
    "ecg2_else_ecg1",
    "limb_preferred",
    "second",
)

# Limb / lead-II style names (checked before defaulting to channel 0).
_LIMB_NAME_MARKERS: Tuple[str, ...] = (
    "MLII",
    "LEADII",
    "LEAD II",
    "ECG1",
)


def _upper_names(sig_names: List[str]) -> List[str]:
    return [str(n).upper() for n in sig_names]


def _index_matching(sig_names: List[str], predicate) -> Optional[int]:
    for i, name in enumerate(sig_names):
        if predicate(_upper_names(sig_names)[i]):
            return i
    return None


def _limb_preferred_index(sig_names: List[str]) -> int:
    """Prefer limb lead II (MLII, etc.); fall back to index 0."""
    upper = _upper_names(sig_names)
    for marker in _LIMB_NAME_MARKERS:
        for i, name in enumerate(upper):
            if marker in name:
                return i
    # Standalone "II" (not already caught as MLII)
    for i, name in enumerate(upper):
        if name == "II" or name.endswith(" II") or name.startswith("II "):
            return i
    return 0


def pick_lead_index(
    sig_names: List[str],
    policy: LeadPolicy = "ecg2_else_ecg1",
) -> int:
    """
    Choose WFDB channel index for analysis.

    Policies
    --------
    first
        Always channel 0 (legacy benchmark behavior).
    second
        Channel 1 when present, else 0 (e.g. V5-only experiments).
    ecg2_else_ecg1
        Channel named ECG2 if present, else ECG1, else limb-preferred, else 0.
    limb_preferred
        MLII / II / ECG1-style names, else 0.
    """
    names = list(sig_names)
    if not names:
        return 0

    if policy == "first":
        return 0
    if policy == "second":
        return 1 if len(names) > 1 else 0
    if policy == "limb_preferred":
        return _limb_preferred_index(names)
    if policy == "ecg2_else_ecg1":
        idx = _index_matching(names, lambda u: "ECG2" in u)
        if idx is not None:
            return idx
        idx = _index_matching(names, lambda u: "ECG1" in u)
        if idx is not None:
            return idx
        return _limb_preferred_index(names)

    raise ValueError(
        f"Unknown lead policy {policy!r}; expected one of {LEAD_POLICIES}"
    )


def pick_manual_annotation_ext(
    sig_names: List[str],
    policy: LeadPolicy = "ecg2_else_ecg1",
    *,
    record_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    QTDB expert annotation extension aligned with :func:`pick_lead_index`.

    Channel 0 → ``q1c``; channel 1 → ``q2c``. When ``record_path`` is given,
    falls back to an existing ``q1c`` / ``q2c`` file if the preferred ext is missing.
    """
    preferred = "q1c" if pick_lead_index(sig_names, policy) == 0 else "q2c"
    if record_path is None:
        return preferred
    base = Path(record_path)
    stem = base.name
    parent = base.parent
    for ext in (preferred, "q1c", "q2c"):
        if (parent / f"{stem}.{ext}").exists():
            return ext
    return preferred


def load_wfdb_signal(
    record_path: Union[str, Path],
    policy: LeadPolicy = "ecg2_else_ecg1",
) -> Tuple[np.ndarray, float, int, str]:
    """
    Load one WFDB record channel.

    Returns
    -------
    signal : np.ndarray
        1-D ECG in physical units (typically mV).
    fs : float
        Sampling frequency (Hz).
    lead_index : int
        Index into ``rec.sig_name``.
    lead_name : str
        Channel name from the header.
    """
    import wfdb

    path = str(record_path)
    rec = wfdb.rdrecord(path)
    lead_index = pick_lead_index(list(rec.sig_name), policy)
    lead_name = rec.sig_name[lead_index]
    sig, _ = wfdb.rdsamp(path, channel_names=[lead_name])
    return np.asarray(sig, dtype=float).flatten(), float(rec.fs), lead_index, str(lead_name)


def load_manual_pt_peaks(
    record_path: Union[str, Path],
    policy: LeadPolicy = "ecg2_else_ecg1",
    *,
    ann_ext: Optional[str] = None,
) -> Tuple[List[int], List[int], str, int]:
    """
    Load manual P and T peak sample indices from QTDB-style annotations.

    Returns
    -------
    p_samples, t_samples, ann_ext_used, lead_index
    """
    import wfdb

    path = str(record_path)
    rec = wfdb.rdrecord(path)
    sig_names = list(rec.sig_name)
    lead_index = pick_lead_index(sig_names, policy)
    ext = ann_ext or pick_manual_annotation_ext(sig_names, policy, record_path=path)
    ann = wfdb.rdann(path, ext)
    p_samples = [int(ann.sample[i]) for i, s in enumerate(ann.symbol) if s == "p"]
    t_samples = [int(ann.sample[i]) for i, s in enumerate(ann.symbol) if s == "t"]
    return p_samples, t_samples, ext, lead_index
