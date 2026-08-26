"""Rebuild a continuous ECG from per-beat Gaussian morphology features.

PyHEARTS stores fitted symmetric Gaussians (center, height, σ) for P/Q/R/S/T on
each retained cycle. This module evaluates those components on a global sample
(or time) axis so beats sit at the same relative locations as in the recording,
then optionally adds residual noise from the original trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .gaussian import gaussian_function

DEFAULT_WAVES: tuple[str, ...] = ("P", "Q", "R", "S", "T")
FWHM_TO_STD = 2.0 * np.sqrt(2.0 * np.log(2.0))
_MIN_WIDTH = 1e-10
_SUPPORT_SIGMAS = 6.0


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _col(row: Mapping[str, object] | pd.Series, name: str) -> object:
    if isinstance(row, pd.Series):
        return row[name] if name in row.index else None
    return row.get(name) if hasattr(row, "get") else None


def _float(row: Mapping[str, object] | pd.Series, name: str) -> float:
    value = _col(row, name)
    return float(value) if _finite(value) else np.nan


def _wave_global_anchor(row: Mapping[str, object] | pd.Series, wave: str) -> float:
    """Refined global sample index used to map cycle-relative Gaussian μ."""
    if wave == "T":
        for name in ("T_gaussian_global_center_idx", "T_global_center_idx"):
            value = _float(row, name)
            if np.isfinite(value):
                return value
    return _float(row, f"{wave}_global_center_idx")


def _wave_height(row: Mapping[str, object] | pd.Series, wave: str) -> float:
    for name in (f"{wave}_gauss_height", f"{wave}_center_voltage"):
        value = _float(row, name)
        if np.isfinite(value):
            return value
    return np.nan


def _wave_sigma(
    row: Mapping[str, object] | pd.Series,
    wave: str,
    sampling_rate: float,
) -> float:
    stdev = _float(row, f"{wave}_gauss_stdev_samples")
    if np.isfinite(stdev) and stdev > 0:
        return float(stdev)

    fwhm = _float(row, f"{wave}_gauss_fwhm_samples")
    if np.isfinite(fwhm) and fwhm > 0:
        return float(fwhm) / FWHM_TO_STD

    stdev_ms = _float(row, f"{wave}_gauss_stdev_ms")
    if np.isfinite(stdev_ms) and stdev_ms > 0:
        return float(stdev_ms) * sampling_rate / 1000.0

    fwhm_ms = _float(row, f"{wave}_gauss_fwhm_ms")
    if np.isfinite(fwhm_ms) and fwhm_ms > 0:
        return float(fwhm_ms) * sampling_rate / 1000.0 / FWHM_TO_STD

    rise_ms = _float(row, f"{wave}_rise_ms")
    decay_ms = _float(row, f"{wave}_decay_ms")
    if np.isfinite(rise_ms) and np.isfinite(decay_ms) and (rise_ms + decay_ms) > 0:
        return (rise_ms + decay_ms) / 4.0 * sampling_rate / 1000.0
    return np.nan


def _wave_mu_samples(
    row: Mapping[str, object] | pd.Series,
    wave: str,
    sampling_rate: float,
    cycle_start: float | None,
) -> float:
    """Gaussian μ on the global sample axis.

    Prefers ``cycle_start + gauss_center`` (the morphology-fit x-axis) when
    segmented cycles are available. Otherwise maps cycle-relative μ through
    the refined global center: ``anchor + gauss_center - center_idx``.
    Time columns are used only when sample indices are missing.
    """
    gauss_center = _float(row, f"{wave}_gauss_center")
    if cycle_start is not None and np.isfinite(gauss_center):
        return float(cycle_start) + float(gauss_center)

    anchor = _wave_global_anchor(row, wave)
    center_idx = _float(row, f"{wave}_center_idx")
    if np.isfinite(gauss_center) and np.isfinite(anchor) and np.isfinite(center_idx):
        return float(anchor) + float(gauss_center) - float(center_idx)
    if np.isfinite(anchor):
        return float(anchor)

    center_ms = _float(row, f"{wave}_center_ms")
    if not np.isfinite(center_ms):
        return np.nan
    mu_ms = center_ms
    if np.isfinite(gauss_center) and np.isfinite(center_idx):
        mu_ms = center_ms + (gauss_center - center_idx) * 1000.0 / sampling_rate
    return mu_ms * sampling_rate / 1000.0


def _edge_samples(
    row: Mapping[str, object] | pd.Series,
    wave: str,
    sampling_rate: float,
    kind: str,
    cycle_start: float | None,
) -> float:
    global_name = f"{wave}_global_{kind}_idx"
    value = _float(row, global_name)
    if np.isfinite(value):
        return value
    rel = _float(row, f"{wave}_{kind}_idx")
    if cycle_start is not None and np.isfinite(rel):
        return float(cycle_start) + float(rel)
    ms = _float(row, f"{wave}_{kind}_ms")
    if np.isfinite(ms):
        return ms * sampling_rate / 1000.0
    return np.nan


def _cycle_start_lookup(
    features: pd.DataFrame,
    cycles: pd.DataFrame | None,
) -> dict[int, float]:
    """Map feature-row position → first global sample of that cycle window."""
    if cycles is None or cycles.empty or "cycle" not in cycles.columns:
        return {}
    if "index" in cycles.columns:
        index_col = "index"
    elif "signal_x" in cycles.columns:
        index_col = "signal_x"
    else:
        return {}

    labels = np.sort(pd.unique(cycles["cycle"]))
    starts: dict[int, float] = {}
    for position, label in enumerate(labels):
        if position >= len(features):
            break
        window = cycles.loc[cycles["cycle"] == label, index_col]
        if window.empty:
            continue
        starts[position] = float(window.iloc[0])
    return starts


def _sigmoid_gate(
    xs: np.ndarray,
    onset: float,
    offset: float,
    sigma: float,
) -> np.ndarray:
    """Keep the Gaussian between onset and offset with smooth edges."""
    scale = 5.0 / max(float(sigma), _MIN_WIDTH)
    left = 1.0 / (1.0 + np.exp(np.clip(-scale * (xs - onset), -500.0, 500.0)))
    right = 1.0 / (1.0 + np.exp(np.clip(scale * (xs - offset), -500.0, 500.0)))
    return left * right


def resolve_wave_params(
    row: Mapping[str, object] | pd.Series,
    wave: str,
    sampling_rate: float,
    cycle_start: float | None = None,
) -> tuple[float, float, float, float, float]:
    """Return ``(mu, height, sigma, onset, offset)`` in samples; NaN if unusable."""
    mu = _wave_mu_samples(row, wave, sampling_rate, cycle_start)
    height = _wave_height(row, wave)
    sigma = _wave_sigma(row, wave, sampling_rate)
    onset = _edge_samples(row, wave, sampling_rate, "le", cycle_start)
    offset = _edge_samples(row, wave, sampling_rate, "ri", cycle_start)
    return mu, height, sigma, onset, offset


def _add_gaussian(
    target: np.ndarray,
    mu: float,
    height: float,
    sigma: float,
    *,
    onset: float | None = None,
    offset: float | None = None,
    gate_edges: bool = False,
) -> None:
    sigma = max(float(sigma), _MIN_WIDTH)
    half = int(np.ceil(_SUPPORT_SIGMAS * sigma))
    i0 = max(0, int(np.floor(mu - half)))
    i1 = min(target.size, int(np.ceil(mu + half)) + 1)
    if i1 <= i0:
        return
    xs = np.arange(i0, i1, dtype=float)
    component = gaussian_function(xs, mu, height, sigma)
    if (
        gate_edges
        and onset is not None
        and offset is not None
        and np.isfinite(onset)
        and np.isfinite(offset)
        and offset > onset
    ):
        component = component * _sigmoid_gate(xs, float(onset), float(offset), sigma)
    target[i0:i1] += component


def _isoelectric_mask(
    n_samples: int,
    params: pd.DataFrame,
    sampling_rate: float,
) -> np.ndarray:
    """True on samples outside every wave's onset–offset (or ±3σ) support."""
    mask = np.ones(n_samples, dtype=bool)
    fallback_ms = 40.0
    pad = max(1, int(round(fallback_ms * sampling_rate / 1000.0)))
    for row in params.itertuples(index=False):
        onset = float(row.onset)
        offset = float(row.offset)
        if not (np.isfinite(onset) and np.isfinite(offset) and offset > onset):
            mu = float(row.mu)
            sigma = float(row.sigma)
            if not (np.isfinite(mu) and np.isfinite(sigma)):
                continue
            onset = mu - 3.0 * sigma
            offset = mu + 3.0 * sigma
        i0 = max(0, int(np.floor(onset)) - pad)
        i1 = min(n_samples, int(np.ceil(offset)) + pad + 1)
        mask[i0:i1] = False
    return mask


def _synthesize_rmse_noise(
    n_samples: int,
    features: pd.DataFrame,
    sampling_rate: float,
    cycle_starts: dict[int, float],
    rng: np.random.Generator,
) -> np.ndarray:
    if "rmse" not in features.columns:
        return np.zeros(n_samples, dtype=float)

    noise = np.zeros(n_samples, dtype=float)
    counts = np.zeros(n_samples, dtype=float)
    global_sigma = float(np.nanmedian(pd.to_numeric(features["rmse"], errors="coerce")))
    if not np.isfinite(global_sigma) or global_sigma < 0:
        global_sigma = 0.0

    for row_number, row in features.iterrows():
        rmse = _float(row, "rmse")
        sigma = rmse if np.isfinite(rmse) and rmse > 0 else global_sigma
        if sigma <= 0:
            continue
        cycle_start = cycle_starts.get(int(row_number))
        left = np.inf
        right = -np.inf
        for wave in DEFAULT_WAVES:
            mu, _, wave_sigma, onset, offset = resolve_wave_params(
                row, wave, sampling_rate, cycle_start
            )
            if np.isfinite(onset):
                left = min(left, onset)
            elif np.isfinite(mu) and np.isfinite(wave_sigma):
                left = min(left, mu - _SUPPORT_SIGMAS * wave_sigma)
            if np.isfinite(offset):
                right = max(right, offset)
            elif np.isfinite(mu) and np.isfinite(wave_sigma):
                right = max(right, mu + _SUPPORT_SIGMAS * wave_sigma)
        if not (np.isfinite(left) and np.isfinite(right) and right > left):
            continue
        i0 = max(0, int(np.floor(left)))
        i1 = min(n_samples, int(np.ceil(right)) + 1)
        if i1 <= i0:
            continue
        noise[i0:i1] += rng.normal(0.0, sigma, size=i1 - i0)
        counts[i0:i1] += 1.0

    covered = counts > 0
    noise[covered] /= counts[covered]
    uncovered = ~covered
    if global_sigma > 0 and np.any(uncovered):
        noise[uncovered] = rng.normal(0.0, global_sigma, size=int(uncovered.sum()))
    return noise


@dataclass
class ReconstructedECG:
    """Global-axis reconstruction of fitted P/Q/R/S/T Gaussians.

    Attributes
    ----------
    index
        Sample index for each point (the morphology-fit x-axis).
    time_ms
        ``index / sampling_rate * 1000``.
    gaussian
        Sum of placed wave Gaussians (clean morphology).
    noise
        Residual of ``original - gaussian`` when an original trace is supplied,
        otherwise RMSE-scaled synthetic residual (or zeros).
    signal
        ``gaussian + noise`` when ``add_noise`` is True, else ``gaussian``.
    components
        Per-wave traces (same length as ``gaussian``).
    params
        One row per placed component with global μ, height, and σ in samples.
    sampling_rate
        Sampling rate in Hz used for time conversion.
    """

    index: np.ndarray
    time_ms: np.ndarray
    gaussian: np.ndarray
    noise: np.ndarray
    signal: np.ndarray
    components: dict[str, np.ndarray]
    params: pd.DataFrame
    sampling_rate: float
    extras: dict[str, object] = field(default_factory=dict)


def reconstruct_cycle(
    row: Mapping[str, object] | pd.Series,
    n_samples: int,
    *,
    sampling_rate: float = 1.0,
    waves: Iterable[str] = DEFAULT_WAVES,
    gate_edges: bool = False,
) -> np.ndarray:
    """Evaluate one beat's Gaussians on the cycle-relative sample axis.

    ``{wave}_gauss_center`` is already in cycle-relative samples, matching
    the x-axis used by the morphology ``curve_fit``. ``sampling_rate`` is
    only needed when σ must be recovered from millisecond columns.
    """
    n_samples = int(n_samples)
    if n_samples <= 0:
        return np.zeros(0, dtype=float)
    fit = np.zeros(n_samples, dtype=float)
    for wave in waves:
        mu = _float(row, f"{wave}_gauss_center")
        height = _wave_height(row, wave)
        sigma = _wave_sigma(row, wave, sampling_rate)
        if not (np.isfinite(mu) and np.isfinite(height) and np.isfinite(sigma) and sigma > 0):
            continue
        onset = _float(row, f"{wave}_le_idx")
        offset = _float(row, f"{wave}_ri_idx")
        _add_gaussian(
            fit,
            mu,
            height,
            sigma,
            onset=onset,
            offset=offset,
            gate_edges=gate_edges,
        )
    return fit


def reconstruct_ecg(
    features: pd.DataFrame,
    sampling_rate: float,
    *,
    original: np.ndarray | None = None,
    cycles: pd.DataFrame | None = None,
    n_samples: int | None = None,
    waves: Iterable[str] = DEFAULT_WAVES,
    add_noise: bool = True,
    noise_mode: str = "residual",
    gate_edges: bool = False,
    rng: np.random.Generator | int | None = None,
) -> ReconstructedECG:
    """Reconstruct a timeseries from per-cycle Gaussian morphology features.

    Each retained P/Q/R/S/T component is rebuilt as a symmetric Gaussian
    (fitted μ, height, σ, with voltage / FWHM / rise–decay fallbacks) and
    added onto a global sample index so beats keep their recorded spacing.
    For T, ``T_gaussian_global_center_idx`` is preferred over record-level
    ``T_global_center_idx``.

    When ``original`` is provided, the residual ``original - gaussian`` is
    captured as noise and (by default) added back so the reconstructed
    signal includes the recording's leftover jitter and baseline. Without
    an original trace, noise can be synthesized from per-cycle ``rmse``.

    Parameters
    ----------
    features
        One row per cardiac cycle (the ``analyze_ecg`` feature table, or a
        saved ``*_pyhearts.csv``).
    sampling_rate
        Sampling rate in Hz.
    original
        Optional 1-D ECG used to estimate residual noise and to set the
        output length. Use the same polarity-corrected trace that was
        analyzed when auto-polarity may have flipped the signal.
    cycles
        Optional epoch table from ``analyze_ecg`` (columns ``index``,
        ``cycle``). When present, cycle-relative ``gauss_center`` is mapped
        as ``index[0] + gauss_center``.
    n_samples
        Output length in samples. Defaults to ``len(original)`` when that
        is given, otherwise spans the placed components from sample 0.
    waves
        Component labels to include. Default P, Q, R, S, T.
    add_noise
        If True (default), ``signal`` is ``gaussian + noise``.
    noise_mode
        ``"residual"`` — ``original - gaussian`` (falls back to RMSE
        synthesis if no original is given);
        ``"isoelectric"`` — residual only outside wave support;
        ``"rmse"`` — synthetic noise scaled by per-cycle ``rmse``.
    gate_edges
        If True, taper each Gaussian with sigmoid onset/offset gates from
        ``*_global_le_idx`` / ``*_global_ri_idx``. Default False matches
        the ungated sum used for ``r_squared`` / ``rmse``.
    rng
        Seed or ``numpy.random.Generator`` for RMSE synthesis.

    Returns
    -------
    ReconstructedECG
        Global-axis Gaussian mixture, residual noise, and their sum.
    """
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if noise_mode not in {"residual", "isoelectric", "rmse"}:
        raise ValueError("noise_mode must be 'residual', 'isoelectric', or 'rmse'")

    table = features.copy()
    if table.index.name == "cycle_index" and "cycle_index" not in table.columns:
        table = table.reset_index()
    table = table.reset_index(drop=True)
    wave_names = tuple(waves)
    cycle_starts = _cycle_start_lookup(table, cycles)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    param_rows: list[dict[str, object]] = []
    max_extent = -1.0
    for row_number, row in table.iterrows():
        cycle_start = cycle_starts.get(int(row_number))
        for wave in wave_names:
            mu, height, sigma, onset, offset = resolve_wave_params(
                row, wave, sampling_rate, cycle_start
            )
            if not (
                np.isfinite(mu)
                and np.isfinite(height)
                and np.isfinite(sigma)
                and sigma > 0
            ):
                continue
            param_rows.append(
                {
                    "cycle_index": int(row_number),
                    "wave": wave,
                    "mu": float(mu),
                    "height": float(height),
                    "sigma": float(sigma),
                    "onset": float(onset) if np.isfinite(onset) else np.nan,
                    "offset": float(offset) if np.isfinite(offset) else np.nan,
                }
            )
            max_extent = max(max_extent, mu + _SUPPORT_SIGMAS * sigma)
            if np.isfinite(offset):
                max_extent = max(max_extent, offset)

    orig = None if original is None else np.asarray(original, dtype=float)
    if orig is not None and orig.ndim != 1:
        raise ValueError("original must be a one-dimensional array")

    if n_samples is not None:
        length = int(n_samples)
    elif orig is not None:
        length = int(orig.size)
    elif max_extent >= 0:
        length = int(np.ceil(max_extent)) + 1
    else:
        length = 0
    if length < 0:
        raise ValueError("n_samples must be non-negative")

    gaussian = np.zeros(length, dtype=float)
    components = {wave: np.zeros(length, dtype=float) for wave in wave_names}
    params = pd.DataFrame(param_rows)

    for row in params.itertuples(index=False):
        if 0 <= float(row.mu) < length or (
            float(row.mu) + _SUPPORT_SIGMAS * float(row.sigma) > 0
            and float(row.mu) - _SUPPORT_SIGMAS * float(row.sigma) < length
        ):
            _add_gaussian(
                components[str(row.wave)],
                float(row.mu),
                float(row.height),
                float(row.sigma),
                onset=float(row.onset),
                offset=float(row.offset),
                gate_edges=gate_edges,
            )
    for wave in wave_names:
        gaussian += components[wave]

    if orig is not None:
        aligned = np.full(length, np.nan, dtype=float)
        n_copy = min(length, orig.size)
        aligned[:n_copy] = orig[:n_copy]
        residual = aligned - gaussian
        residual[~np.isfinite(residual)] = 0.0
    else:
        residual = None

    if noise_mode == "rmse" or (residual is None and noise_mode in {"residual", "isoelectric"}):
        noise = _synthesize_rmse_noise(
            length, table, sampling_rate, cycle_starts, generator
        )
        noise_source = "rmse"
    elif residual is None:
        noise = np.zeros(length, dtype=float)
        noise_source = "none"
    elif noise_mode == "isoelectric":
        mask = _isoelectric_mask(length, params, sampling_rate)
        noise = np.zeros(length, dtype=float)
        noise[mask] = residual[mask]
        iso = residual[mask]
        iso = iso[np.isfinite(iso)]
        if iso.size:
            fill_sigma = float(np.median(np.abs(iso - np.median(iso))) * 1.4826)
            if fill_sigma > 0:
                noise[~mask] = generator.normal(0.0, fill_sigma, size=int((~mask).sum()))
        noise_source = "isoelectric"
    else:
        noise = residual
        noise_source = "residual"

    signal = gaussian + noise if add_noise else gaussian.copy()
    index = np.arange(length, dtype=float)
    return ReconstructedECG(
        index=index,
        time_ms=index * 1000.0 / sampling_rate,
        gaussian=gaussian,
        noise=np.asarray(noise, dtype=float),
        signal=signal,
        components=components,
        params=params,
        sampling_rate=float(sampling_rate),
        extras={"noise_source": noise_source, "n_components": int(len(params))},
    )
