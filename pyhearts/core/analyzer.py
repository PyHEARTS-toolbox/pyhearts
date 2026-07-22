"""Public PyHEARTS analyzer (morphology core + record-level T).

Typical usage constructs :class:`PyHEARTS` with ``species=`` and calls
:meth:`PyHEARTS.analyze_ecg`. Dual morphology/record-T configs stay private.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyhearts._morphology import (
    PyHEARTS as _CorePyHEARTS,
)
from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import prepare_record_delineation_signal
from pyhearts.processing.record_delineation import (
    _local_rr_samples,
    _resolve_t_guess,
    build_record_beat_template,
    delineate_record_template,
)
from pyhearts.version import __version__

# Algorithm tag recorded in saved metadata (distinct from the package version).
PIPELINE_VERSION = "morphology-record-t"


def detect_record_t(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Detect record-level T centers for each R peak.

    Builds a beat template from the recording, then projects a T-wave guess
    onto each cardiac cycle. Used internally at the end of
    :meth:`PyHEARTS.analyze_ecg` when record-T is enabled.

    Parameters
    ----------
    ecg_signal : np.ndarray
        One-dimensional ECG (same units/sampling as used for morphology fitting).
    r_peaks : np.ndarray
        R-peak sample indices (finite values only are used).
    sampling_rate : float
        Sampling rate in Hz (must be positive).
    cfg : ProcessCycleConfig, optional
        record-T/delineation settings. Defaults to the human-unified record-T preset.

    Returns
    -------
    pairs : np.ndarray
        Shape ``(n_r, 2)`` array of ``[R_sample, T_sample]``. ``T_sample`` is
        ``NaN`` when detection fails for that beat.
    stats : dict
        Diagnostic counters (template validity, detections, misses, sources).
    """

    ecg = np.asarray(ecg_signal, dtype=float)
    if ecg.ndim != 1:
        raise ValueError("ecg_signal must be a one-dimensional array")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    t_cfg = cfg or ProcessCycleConfig.for_human_unified()
    t_cfg = replace(
        t_cfg,
        record_delineation=True,
        record_delineation_replace_p=False,
        record_delineation_overwrite_existing_p=False,
        record_delineation_map_p_even_if_finite=False,
        version=PIPELINE_VERSION,
    )

    r_values = np.asarray(r_peaks, dtype=float)
    r_values = r_values[np.isfinite(r_values)]
    r_values = np.unique(np.round(r_values).astype(int))
    r_values = r_values[(r_values >= 0) & (r_values < ecg.size)]

    stats: dict[str, Any] = {
        "n_r": int(r_values.size),
        "template_valid": 0,
        "t_detected": 0,
        "t_record_miss": 0,
        "t_template_fallback": 0,
    }
    if r_values.size < 2:
        return np.column_stack([r_values.astype(float), np.full(r_values.size, np.nan)]), stats

    delineation_ecg = prepare_record_delineation_signal(ecg, sampling_rate, t_cfg)
    raw_template = build_record_beat_template(ecg, r_values, sampling_rate, t_cfg)
    template = delineate_record_template(raw_template, sampling_rate, t_cfg)
    if not template.valid:
        return np.column_stack([r_values.astype(float), np.full(r_values.size, np.nan)]), stats
    stats["template_valid"] = 1

    cycle_labels = np.arange(r_values.size)
    pairs: list[tuple[float, float]] = []
    for cycle_idx, r_detection in enumerate(r_values):
        r_next = int(r_values[cycle_idx + 1]) if cycle_idx + 1 < r_values.size else None
        local_rr = _local_rr_samples(
            cycle_idx,
            r_values,
            cycle_labels,
            template.median_rr_samples,
        )
        scale = 1.0
        if t_cfg.record_delineation_rr_scale_pt and template.median_rr_samples > 0:
            low, high = t_cfg.record_delineation_rr_scale_bounds
            scale = float(np.clip(local_rr / template.median_rr_samples, low, high))

        guess_stats: dict[str, int] = {}
        t_guess, source = _resolve_t_guess(
            ecg_delim=delineation_ecg,
            r_det=int(r_detection),
            r_next=r_next,
            r_g=float(r_detection),
            tmpl=template,
            sampling_rate=float(sampling_rate),
            cfg=t_cfg,
            scale=scale,
            stats=guess_stats,
        )
        for key, value in guess_stats.items():
            stats[key] = stats.get(key, 0) + int(value)

        if t_guess is None:
            stats["t_record_miss"] = stats.get("t_record_miss", 0) + 1
            pairs.append((float(r_detection), np.nan))
            continue

        stats["t_detected"] += 1
        if source:
            source_key = f"source_{source}"
            stats[source_key] = stats.get(source_key, 0) + 1
        pairs.append((float(r_detection), float(t_guess)))

    return np.asarray(pairs, dtype=float), stats


def merge_record_t(
    features: pd.DataFrame,
    pairs: np.ndarray,
    *,
    max_r_distance_samples: int = 40,
) -> pd.DataFrame:
    """
    Merge record-level T detections into a morphology feature table.

    Preserves Gaussian morphology by copying the pre-merge
    ``T_global_center_idx`` into ``T_gaussian_global_center_idx``, then
    overwrites only ``T_global_center_idx`` from nearest-matched record-T pairs.

    Parameters
    ----------
    features : pandas.DataFrame
        Per-cycle feature table from morphology fitting. Must include
        ``R_global_center_idx`` for matching.
    pairs : np.ndarray
        ``[R, T]`` pairs from :func:`detect_record_t`.
    max_r_distance_samples : int, default 40
        Maximum absolute sample distance allowed when matching record-T R indices
        to morphology R indices.

    Returns
    -------
    pandas.DataFrame
        Copy of ``features`` with updated ``T_global_center_idx``,
        ``T_gaussian_global_center_idx``, and ``t_source``
        (``record_t``, ``record_t_miss``, or ``missing``).
    """

    output = features.copy()
    if "T_global_center_idx" not in output:
        output["T_global_center_idx"] = np.nan
    output["T_gaussian_global_center_idx"] = pd.to_numeric(
        output["T_global_center_idx"], errors="coerce"
    )
    if "R_global_center_idx" not in output or pairs.size == 0:
        output["t_source"] = "missing"
        return output

    detected_r = pd.to_numeric(output["R_global_center_idx"], errors="coerce").to_numpy(dtype=float)
    record_t_r = pairs[:, 0]
    record_t_t = pairs[:, 1]
    assigned = np.full(len(output), np.nan, dtype=float)
    sources = np.full(len(output), "missing", dtype=object)
    used: set[int] = set()

    for cycle_idx, r_value in enumerate(detected_r):
        if not np.isfinite(r_value):
            continue
        distances = np.abs(record_t_r - r_value)
        for pair_idx in np.argsort(distances):
            pair_idx = int(pair_idx)
            if pair_idx in used:
                continue
            if distances[pair_idx] > max_r_distance_samples:
                break
            used.add(pair_idx)
            if np.isfinite(record_t_t[pair_idx]):
                assigned[cycle_idx] = record_t_t[pair_idx]
                sources[cycle_idx] = "record_t"
            else:
                sources[cycle_idx] = "record_t_miss"
            break

    output["T_global_center_idx"] = assigned
    output["t_source"] = sources
    return output


class PyHEARTS(_CorePyHEARTS):
    """
    Beat-by-beat ECG morphology analyzer with optional record-level T.

    The public workflow is:

    1. Construct with ``sampling_rate`` and ``species``.
    2. Optionally call :meth:`preprocess_signal`.
    3. Call :meth:`analyze_ecg` to obtain per-beat features and cycle traces.
    4. Optionally call :meth:`save_output` to write CSV + metadata.

    Species presets
    ---------------
    ``None``
        Species-agnostic morphology defaults (SPH-style validation setting);
        record-level T enabled.
    ``"human"``
        Human morphology preset; record-level T enabled.
    ``"mouse"``
        Mouse morphology preset; record-level T disabled by default.

    Notes
    -----
    Morphology and record-T configs are private (``_core_cfg``, ``_t_cfg``).
    Inspect ``pipeline_version`` for the algorithm tag and
    ``pyhearts.__version__`` for the package version.
    """

    def __init__(
        self,
        sampling_rate: float,
        verbose: bool = False,
        plot: bool = False,
        *,
        species: str | None = None,
        **overrides: Any,
    ):
        """
        Create a PyHEARTS analyzer.

        Parameters
        ----------
        sampling_rate : float
            ECG sampling rate in Hz (must be positive).
        verbose : bool, default False
            If True, print progress during analysis.
        plot : bool, default False
            If True, enable diagnostic plots during analysis when supported.
        species : {None, "human", "mouse"}, optional
            Morphology/record-T preset selector. See the class docstring.
        **overrides
            Optional morphology-core field overrides forwarded to the
            underlying analyzer. Prefer ``species=`` for normal use.

        Raises
        ------
        ValueError
            If ``sampling_rate`` is not positive or ``species`` is invalid.
        """
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if species not in (None, "human", "mouse"):
            raise ValueError("species must be None, 'human', or 'mouse'")

        super().__init__(
            sampling_rate=float(sampling_rate),
            verbose=bool(verbose),
            plot=bool(plot),
            species=species,
            **overrides,
        )
        self.species = species
        # ``self.cfg`` is the morphology-core config used by inherited methods.
        self._core_cfg = self.cfg
        self._t_cfg = replace(
            ProcessCycleConfig.for_human_unified(),
            version=PIPELINE_VERSION,
        )
        self.apply_record_t = species != "mouse"
        self.last_record_t_stats: dict[str, Any] = {}
        self.pipeline_version = PIPELINE_VERSION

    def analyze_ecg(
        self,
        ecg_signal: np.ndarray,
        verbose: bool | None = None,
        plot: bool | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit cardiac-cycle morphology, then apply record-level T when enabled.

        Morphology detection segments beats and fits symmetric Gaussians for
        P/Q/R/S/T. When record-T is enabled (human / species-agnostic presets), a
        record-level record-T stage overwrites ``T_global_center_idx`` while keeping
        the Gaussian T center in ``T_gaussian_global_center_idx``.

        Parameters
        ----------
        ecg_signal : np.ndarray
            One-dimensional ECG. Preprocess first when the recording needs
            filtering (see :meth:`preprocess_signal`).
        verbose : bool, optional
            Override the instance ``verbose`` flag for this call.
        plot : bool, optional
            Override the instance ``plot`` flag for this call.
        **kwargs
            Additional arguments forwarded to the morphology analyzer.

        Returns
        -------
        features : pandas.DataFrame
            One row per cardiac cycle (fiducials, Gaussian morphology,
            intervals, fit quality). Key T columns include
            ``T_global_center_idx``, ``T_gaussian_global_center_idx``, and
            ``t_source``.
        cycles : pandas.DataFrame
            Segmented per-beat waveform samples.

        Notes
        -----
        Results are also stored on the instance as ``output_df`` / ``epochs_df``
        for :meth:`save_output`.
        """
        features, epochs = super().analyze_ecg(
            np.asarray(ecg_signal, dtype=float),
            verbose=verbose,
            plot=plot,
            **kwargs,
        )
        output = features.copy()

        if self.apply_record_t and not output.empty:
            r_values = pd.to_numeric(output.get("R_global_center_idx"), errors="coerce").to_numpy(
                dtype=float
            )
            pairs, self.last_record_t_stats = detect_record_t(
                np.asarray(ecg_signal, dtype=float),
                r_values,
                self.sampling_rate,
                self._t_cfg,
            )
            output = merge_record_t(output, pairs)
            self.output_df = output
            self.output_dict = output.to_dict(orient="list")

        self.epochs_df = epochs
        return output, epochs

    def save_output(self, file_id: str, results_dir: str):
        """
        Write the latest feature table and reproducibility metadata to disk.

        Requires a prior successful :meth:`analyze_ecg` call so ``output_df``
        is populated.

        Parameters
        ----------
        file_id : str
            Stem used for output filenames (for example ``"subject_001"``).
        results_dir : str
            Directory to create if needed and write into.

        Returns
        -------
        pathlib.Path
            Path to the written features CSV
            (``{file_id}_pyhearts.csv``).

        Notes
        -----
        Also writes ``{file_id}_meta.json`` containing package version,
        pipeline tag, species, record-T flag, resolved configs, and runtime info.
        """
        output_dir = Path(results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{file_id}_pyhearts.csv"
        self.output_df.to_csv(output_path, index=True, na_rep="NaN")
        metadata = {
            "pyhearts_version": __version__,
            "pipeline_version": self.pipeline_version,
            "pipeline": {
                "morphology": "symmetric-gaussian",
                "t_detector": "record-t",
                "t_output": "T_global_center_idx",
            },
            "sampling_rate_hz": self.sampling_rate,
            "species": self.species,
            "apply_record_t": self.apply_record_t,
            "core_config": asdict(self._core_cfg),
            "t_config": asdict(self._t_cfg),
            "last_record_t_stats": self.last_record_t_stats,
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        (output_dir / f"{file_id}_meta.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )
        return output_path


__all__ = [
    "PIPELINE_VERSION",
    "PyHEARTS",
    "detect_record_t",
    "merge_record_t",
]
