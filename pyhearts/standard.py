"""
Standardized PyHEARTS pipeline for multi-cohort ECG analysis.

Every cohort runner (and the LUDB held-out benchmark) calls :func:`run_standard`
so that lead selection, sampling-rate checks, missing-sample handling,
filtering, analysis, HRV and output metadata are identical across datasets.
Cohort scripts only load data and declare site facts (expected sampling rate,
mains frequency, lead layout).

Steps
-----
1. Lead II only (:func:`find_lead_ii`). Records without lead II are excluded.
2. Sampling rate must equal the dataset's documented rate
   (:func:`check_sampling_rate`).
3. Missing samples (non-finite values) are never analyzed. Gaps are linearly
   bridged only so the zero-phase filters can run; every beat and RR interval
   within ``guard_s`` of a missing sample is then excluded. Records with more
   than ``max_missing_frac`` missing samples are excluded.
4. Filtering: 0.5 Hz high-pass, 50 Hz low-pass (human; 500 Hz mouse), both
   4th-order Butterworth applied forward-backward (zero phase), plus a notch
   (Q = 30) at the site mains frequency. No polynomial detrending.
5. ``PyHEARTS(species=...)`` analysis, HRV from R peaks, and output with the
   preprocessing settings written into ``{record_id}_meta.json``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ._morphology.feature import calc_hrv_metrics
from .core.analyzer import PyHEARTS
from .version import __version__

STANDARD_PIPELINE_VERSION = "standard-2"  # standard-2: unrounded Gaussian SD bounds


@dataclass(frozen=True)
class FilterSettings:
    """Zero-phase Butterworth band limits and notch quality factor."""

    highpass_hz: float = 0.5
    lowpass_hz: float = 50.0
    order: int = 4
    notch_q: float = 30.0


HUMAN_FILTER = FilterSettings()
# Mouse QRS is ~3 ms wide (FWHM); <300 Hz low-pass measurably attenuates it.
MOUSE_FILTER = FilterSettings(lowpass_hz=500.0)


@dataclass(frozen=True)
class MissingDataPolicy:
    """
    Exclusion rules for non-finite samples.

    guard_s
        Beats and RR intervals within this many seconds of a missing sample are
        excluded. 2.0 s covers the standard filter chain's impulse response,
        which decays below 0.1% of its peak within 1.64 s.
    max_missing_frac
        Records with a larger fraction of missing samples are excluded.
    """

    guard_s: float = 2.0
    max_missing_frac: float = 0.20


DEFAULT_MISSING_POLICY = MissingDataPolicy()


class RecordExcluded(Exception):
    """A record failed a standard inclusion check. ``reason`` is a short code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------- lead II ---

_LEAD_II_RE = re.compile(r"^\s*(ECG\s*)?(LEAD\s*)?II\s*$", re.IGNORECASE)


def find_lead_ii(
    sig_names: Sequence[str],
    *,
    single_ecg_channel_is_lead_ii: bool = False,
) -> int:
    """
    Return the channel index of standard limb lead II.

    Accepts names such as ``II``, ``ii``, ``Lead II`` and ``ECG lead II``.
    Modified or derived leads (``MLII``) are not accepted.

    Parameters
    ----------
    sig_names
        Channel names from the record header.
    single_ecg_channel_is_lead_ii
        For datasets whose documentation states the only ECG channel is lead II
        but whose header names it generically (e.g. Autonomic Aging, ``ECG``).

    Raises
    ------
    RecordExcluded
        ``no_lead_ii`` or ``ambiguous_lead_ii``.
    """
    names = [str(n) for n in sig_names]
    matches = [i for i, n in enumerate(names) if _LEAD_II_RE.match(n)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RecordExcluded("ambiguous_lead_ii", f"channels={names}")
    if single_ecg_channel_is_lead_ii:
        ecg_like = [i for i, n in enumerate(names) if "ECG" in n.upper()]
        if len(ecg_like) == 1:
            return ecg_like[0]
    raise RecordExcluded("no_lead_ii", f"channels={names}")


# ---------------------------------------------------------- sampling rate ---


def check_sampling_rate(fs: Any, expected_fs: float, *, tol_hz: float = 1e-6) -> float:
    """Return ``fs`` as float if it equals the documented rate, else exclude."""
    try:
        fs_f = float(fs)
    except (TypeError, ValueError):
        raise RecordExcluded("invalid_sampling_rate", f"fs={fs!r}") from None
    if not np.isfinite(fs_f) or fs_f <= 0:
        raise RecordExcluded("invalid_sampling_rate", f"fs={fs!r}")
    if abs(fs_f - float(expected_fs)) > tol_hz:
        raise RecordExcluded(
            "sampling_rate_mismatch", f"fs={fs_f} Hz, expected {expected_fs} Hz"
        )
    return fs_f


def check_sample_count(n_samples: int, expected_n: int) -> None:
    """For files without a stored rate: length must match the dataset metadata."""
    if int(n_samples) != int(expected_n):
        raise RecordExcluded(
            "sample_count_mismatch", f"n={n_samples}, metadata={expected_n}"
        )


# ----------------------------------------------------------- missing data ---


def _dilate(mask: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or not mask.any():
        return mask.copy()
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    idx = np.arange(mask.size)
    lo = np.clip(idx - n, 0, mask.size)
    hi = np.clip(idx + n + 1, 0, mask.size)
    return (c[hi] - c[lo]) > 0


def bridge_missing(ecg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (signal with gaps linearly bridged, boolean missing mask).

    Bridged samples exist only so the filters can run; callers must exclude
    any beat or interval that touches the (guarded) mask.
    """
    x = np.asarray(ecg, dtype=float).ravel().copy()
    if x.size == 0:
        raise RecordExcluded("empty_signal")
    mask = ~np.isfinite(x)
    if mask.all():
        raise RecordExcluded("all_missing")
    if mask.any():
        idx = np.arange(x.size)
        x[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return x, mask


def _excluded_beat_rows(ph: PyHEARTS, bad: np.ndarray) -> np.ndarray:
    """Boolean array over ``ph.output_df`` rows whose epoch touches ``bad``."""
    df = ph.output_df
    r = pd.to_numeric(df.get("R_global_center_idx"), errors="coerce").to_numpy(dtype=float)
    rp = getattr(ph, "r_peak_indices", None)
    peaks = np.asarray(rp if rp is not None else [], dtype=float)
    if peaks.size >= 2:
        half = 0.5 * float(np.mean(np.diff(np.sort(peaks))))
    else:
        half = 0.5 * float(ph.sampling_rate)  # 0.5 s fallback
    c = np.concatenate([[0], np.cumsum(bad.astype(np.int64))])
    out = np.zeros(len(df), dtype=bool)
    for i, ri in enumerate(r):
        if not np.isfinite(ri):
            out[i] = True
            continue
        lo = int(max(0, np.floor(ri - half)))
        hi = int(min(bad.size, np.ceil(ri + half) + 1))
        out[i] = (c[hi] - c[lo]) > 0
    return out


def _hrv_excluding(ph: PyHEARTS, bad: np.ndarray) -> int:
    """Same HRV as ``compute_hrv_metrics`` but drop RR intervals touching ``bad``."""
    peaks = np.sort(np.unique(np.round(np.asarray(ph.r_peak_indices, float)).astype(int)))
    c = np.concatenate([[0], np.cumsum(bad.astype(np.int64))])
    keep_rr = [
        (c[min(b + 1, bad.size)] - c[max(a, 0)]) == 0 for a, b in zip(peaks[:-1], peaks[1:])
    ]
    rr_ms = np.diff(peaks).astype(float) * (1000.0 / float(ph.sampling_rate))
    n_dropped = int(np.size(keep_rr) - np.sum(keep_rr))
    rr_ms = rr_ms[np.asarray(keep_rr, dtype=bool)]
    lo, hi = getattr(ph.cfg, "rr_bounds_ms", (None, None)) or (None, None)
    if lo is not None:
        rr_ms = rr_ms[(rr_ms >= float(lo)) & (rr_ms <= float(hi))]
    ph.rr_intervals_ms = rr_ms
    if rr_ms.size < 60:
        ph.hrv_metrics = {}
        return n_dropped
    hr, sdnn, rmssd, nn50 = calc_hrv_metrics(rr_ms)
    ph.hrv_metrics = {
        "average_heart_rate": hr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "nn50": nn50,
        "rr_source": "r_peaks",
        "n_r_peaks": int(peaks.size),
        "n_rr_intervals": int(rr_ms.size),
        "n_rr_excluded_missing": n_dropped,
    }
    return n_dropped


def log_exclusion(
    results_dir: str | Path, record_id: str, reason: str, detail: str = ""
) -> None:
    """Append one row to ``{results_dir}/exclusions.csv`` (record_id,reason,detail)."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "exclusions.csv"
    if not path.exists():
        path.write_text("record_id,reason,detail\n", encoding="utf-8")
    clean = str(detail).replace(",", ";").replace("\n", " ")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{record_id},{reason},{clean}\n")


# -------------------------------------------------------------- pipeline ---


def run_standard(
    ecg_raw: np.ndarray,
    fs: Any,
    *,
    expected_fs: float,
    mains_hz: float,
    record_id: str,
    results_dir: Optional[str | Path] = None,
    species: str = "human",
    lead_name: Optional[str] = None,
    dataset: Optional[str] = None,
    missing_policy: MissingDataPolicy = DEFAULT_MISSING_POLICY,
    core_overrides: Optional[dict[str, Any]] = None,
    save_rr_intervals: bool = False,
) -> tuple[PyHEARTS, dict[str, Any]]:
    """
    Run the standardized pipeline on one lead-II signal.

    Returns the fitted analyzer (features in ``ph.output_df``) and a summary
    dict. Raises :class:`RecordExcluded` for records failing inclusion checks.
    When ``results_dir`` is given, writes features, HRV and metadata there.
    """
    if float(mains_hz) not in (50.0, 60.0):
        raise ValueError(f"mains_hz must be 50 or 60; got {mains_hz}")
    if species not in ("human", "mouse"):
        raise ValueError("species must be 'human' or 'mouse'")
    filt = MOUSE_FILTER if species == "mouse" else HUMAN_FILTER

    fs_f = check_sampling_rate(fs, expected_fs)
    if filt.lowpass_hz >= fs_f / 2.0:
        raise ValueError(f"low-pass {filt.lowpass_hz} Hz >= Nyquist for fs={fs_f}")

    x, missing = bridge_missing(ecg_raw)
    missing_frac = float(missing.mean())
    if missing_frac > missing_policy.max_missing_frac:
        raise RecordExcluded(
            "too_much_missing",
            f"{missing_frac:.3f} > {missing_policy.max_missing_frac}",
        )

    ph = PyHEARTS(
        sampling_rate=fs_f,
        species=species,
        verbose=False,
        plot=False,
        **(core_overrides or {}),
    )
    y = ph.preprocess_signal(
        x,
        highpass_cutoff=filt.highpass_hz,
        lowpass_cutoff=filt.lowpass_hz,
        filter_order=filt.order,
        notch_frequency=float(mains_hz),
        quality_factor=filt.notch_q,
    )
    ph.analyze_ecg(y)

    n_beats_total = int(len(ph.output_df)) if getattr(ph, "output_df", None) is not None else 0
    n_beats_excluded = 0
    n_rr_excluded = 0
    if missing.any():
        bad = _dilate(missing, int(round(missing_policy.guard_s * fs_f)))
        if n_beats_total:
            drop = _excluded_beat_rows(ph, bad)
            n_beats_excluded = int(drop.sum())
            ph.output_df = ph.output_df.loc[~drop]
            ph.output_dict = ph.output_df.to_dict(orient="list")
        if getattr(ph, "r_peak_indices", None) is not None and len(ph.r_peak_indices) >= 2:
            n_rr_excluded = _hrv_excluding(ph, bad)
        else:
            ph.hrv_metrics = {}
    else:
        ph.compute_hrv_metrics()

    summary: dict[str, Any] = {
        "standard_pipeline_version": STANDARD_PIPELINE_VERSION,
        "pyhearts_version": __version__,
        "dataset": dataset,
        "record_id": record_id,
        "species": species,
        "lead": lead_name,
        "lead_policy": "lead_II_only",
        "sampling_rate_hz": fs_f,
        "expected_sampling_rate_hz": float(expected_fs),
        "filter": {
            **asdict(filt),
            "type": "butterworth",
            "application": "zero-phase forward-backward (scipy.signal.filtfilt)",
            "notch_hz": float(mains_hz),
            "polynomial_detrend": None,
        },
        "missing_data": {
            **asdict(missing_policy),
            "n_samples": int(missing.size),
            "n_missing": int(missing.sum()),
            "missing_frac": missing_frac,
            "n_beats_before_missing_exclusion": n_beats_total,
            "n_beats_excluded_missing": n_beats_excluded,
            "n_rr_excluded_missing": n_rr_excluded,
        },
        "core_overrides": core_overrides or {},
    }

    if results_dir is not None:
        out = Path(results_dir)
        ph.save_output(record_id, str(out))
        ph.save_hrv_metrics(record_id, str(out))
        if save_rr_intervals:
            ph.save_rr_intervals(record_id, str(out))
        meta_path = out / f"{record_id}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta["standard_pipeline"] = summary
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    return ph, summary


__all__ = [
    "STANDARD_PIPELINE_VERSION",
    "FilterSettings",
    "HUMAN_FILTER",
    "MOUSE_FILTER",
    "MissingDataPolicy",
    "DEFAULT_MISSING_POLICY",
    "RecordExcluded",
    "find_lead_ii",
    "check_sampling_rate",
    "check_sample_count",
    "bridge_missing",
    "run_standard",
    "log_exclusion",
]
