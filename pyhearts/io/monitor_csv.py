from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonitorCSVLoadResult:
    """
    Result of loading a 2-column monitor CSV.

    Notes
    -----
    Many bedside/ambulatory monitor exports round/quantize the displayed timestamp
    (e.g., to 0.1s), so the timestamp column may repeat across many consecutive samples.
    In that case, you should treat the signal as uniformly sampled and use `sampling_rate_hz`.
    """

    ecg: np.ndarray
    sampling_rate_hz: float
    time_s: np.ndarray
    meta: dict[str, Any]


def _parse_minsec_timestamp_to_seconds(series: pd.Series) -> np.ndarray:
    """
    Parse timestamps like 'MM:SS.s' into seconds.

    This is intentionally permissive (accepts 'M:SS', 'MM:SS', 'MM:SS.sss').
    """
    parts = series.astype(str).str.split(":", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("Expected timestamps like 'MM:SS.s' (minute:second.fraction).")
    mins = pd.to_numeric(parts[0], errors="coerce")
    secs = pd.to_numeric(parts[1], errors="coerce")
    out = (mins * 60.0 + secs).to_numpy(dtype=float)
    if np.isnan(out).any():
        raise ValueError("Failed to parse one or more timestamps in the first column.")
    return out


def load_monitor_csv(
    path: Union[str, Path],
    *,
    sampling_rate_hz: Optional[float] = 500.0,
    adc_midpoint: Optional[float] = 8192.0,
    mv_per_count: Optional[float] = None,
    assume_uniform_sampling: bool = True,
) -> MonitorCSVLoadResult:
    """
    Load a 2-column CSV exported from a monitor: (time, value).

    Parameters
    ----------
    path:
        Path to a CSV with two columns: time and signal.
        The file is assumed to be headerless (as in your example).
    sampling_rate_hz:
        Sampling rate (Hz). If None, we attempt to estimate from the first/last timestamps.
        For monitor exports where the timestamp column is rounded (repeats), providing this
        explicitly is recommended.
    adc_midpoint:
        If provided, subtract this midpoint from the raw signal (common for unsigned ADC exports).
        Set to None to leave the raw signal unchanged.
    mv_per_count:
        If provided, convert ADC counts to mV: `ecg_mV = (counts - adc_midpoint) * mv_per_count`.
        If None, the output `ecg` will be in centered counts (or raw units if adc_midpoint=None).
    assume_uniform_sampling:
        If True (default), generate `time_s` from `sampling_rate_hz` and ignore per-row timestamps.
        If False, uses the parsed per-row timestamps directly (requires monotonic per-sample times).

    Returns
    -------
    MonitorCSVLoadResult
        `ecg` is float32, 1D. Units are mV if `mv_per_count` is provided, else counts/unknown.
    """
    path = Path(path)
    df = pd.read_csv(path, header=None, names=["t", "y"])
    if df.shape[1] < 2:
        raise ValueError("Expected at least 2 columns (time, signal).")

    time_s_parsed = _parse_minsec_timestamp_to_seconds(df["t"])
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(y).any():
        raise ValueError("Signal column contains non-numeric values.")

    # Estimate fs from endpoints when possible (even if timestamps are rounded, this is useful)
    span_s = float(time_s_parsed[-1] - time_s_parsed[0])
    fs_est = None
    if span_s > 0:
        fs_est = (len(y) - 1) / span_s

    if sampling_rate_hz is None:
        if fs_est is None:
            raise ValueError("Cannot infer sampling_rate_hz: timestamps have zero span.")
        sampling_rate_hz = float(fs_est)

    # Build signal in desired units
    y_centered = y
    if adc_midpoint is not None:
        y_centered = y_centered - float(adc_midpoint)
    if mv_per_count is not None:
        y_centered = y_centered * float(mv_per_count)

    ecg = y_centered.astype(np.float32, copy=False)

    if assume_uniform_sampling:
        time_s = (np.arange(len(ecg), dtype=np.float64) / float(sampling_rate_hz)) + float(
            time_s_parsed[0]
        )
    else:
        # If the monitor timestamp is rounded, this will contain repeats and violate monotonicity.
        if np.any(np.diff(time_s_parsed) <= 0):
            raise ValueError(
                "Timestamp column is not strictly increasing per sample. "
                "Use assume_uniform_sampling=True and provide sampling_rate_hz."
            )
        time_s = time_s_parsed.astype(np.float64, copy=False)

    meta: dict[str, Any] = {
        "path": str(path),
        "n_samples": int(len(ecg)),
        "fs_est_from_endpoints_hz": None if fs_est is None else float(fs_est),
        "adc_midpoint": None if adc_midpoint is None else float(adc_midpoint),
        "mv_per_count": None if mv_per_count is None else float(mv_per_count),
        "timestamp_start_s": float(time_s_parsed[0]),
        "timestamp_end_s": float(time_s_parsed[-1]),
    }
    return MonitorCSVLoadResult(ecg=ecg, sampling_rate_hz=float(sampling_rate_hz), time_s=time_s, meta=meta)


