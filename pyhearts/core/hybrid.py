"""Validated PyHEARTS hybrid pipeline.

The public analyzer keeps the 2025 R/P detection, cycle segmentation, and
Gaussian morphology fit, then replaces only the global T-wave center with the
record-level STPQ detector from the newer human pipeline.
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

from pyhearts._legacy2025 import (
    ProcessCycleConfig as CoreProcessCycleConfig,
)
from pyhearts._legacy2025 import (
    PyHEARTS as CorePyHEARTS,
)
from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import prepare_record_delineation_signal
from pyhearts.processing.record_delineation import (
    _local_rr_samples,
    _resolve_t_guess,
    build_record_beat_template,
    delineate_record_template,
)

HYBRID_CONFIG_VERSION = "hybrid-t-2025-stpq"


def detect_record_stpq_t(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``[R, T]`` sample pairs from the record-level STPQ detector.

    Parameters
    ----------
    ecg_signal
        One-dimensional filtered ECG used by the 2025 core analysis.
    r_peaks
        R centers produced by the 2025 Gaussian pipeline.
    sampling_rate
        Signal sampling rate in Hz.
    cfg
        Optional STPQ configuration. The production human-unified preset is
        used by default.
    """

    ecg = np.asarray(ecg_signal, dtype=float)
    if ecg.ndim != 1:
        raise ValueError("ecg_signal must be a one-dimensional array")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    stpq_cfg = cfg or ProcessCycleConfig.for_human_unified()
    stpq_cfg = replace(
        stpq_cfg,
        record_delineation=True,
        record_delineation_replace_p=False,
        record_delineation_overwrite_existing_p=False,
        record_delineation_map_p_even_if_finite=False,
        version=HYBRID_CONFIG_VERSION,
    )

    r_values = np.asarray(r_peaks, dtype=float)
    r_values = r_values[np.isfinite(r_values)]
    r_values = np.unique(np.round(r_values).astype(int))
    r_values = r_values[(r_values >= 0) & (r_values < ecg.size)]

    stats: dict[str, Any] = {
        "n_r": int(r_values.size),
        "template_valid": 0,
        "t_detected": 0,
        "t_stpq_miss": 0,
        "t_template_fallback": 0,
    }
    if r_values.size < 2:
        return np.column_stack([r_values.astype(float), np.full(r_values.size, np.nan)]), stats

    delineation_ecg = prepare_record_delineation_signal(ecg, sampling_rate, stpq_cfg)
    raw_template = build_record_beat_template(ecg, r_values, sampling_rate, stpq_cfg)
    template = delineate_record_template(raw_template, sampling_rate, stpq_cfg)
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
        if stpq_cfg.record_delineation_rr_scale_pt and template.median_rr_samples > 0:
            low, high = stpq_cfg.record_delineation_rr_scale_bounds
            scale = float(np.clip(local_rr / template.median_rr_samples, low, high))

        guess_stats: dict[str, int] = {}
        t_guess, source = _resolve_t_guess(
            ecg_delim=delineation_ecg,
            r_det=int(r_detection),
            r_next=r_next,
            r_g=float(r_detection),
            tmpl=template,
            sampling_rate=float(sampling_rate),
            cfg=stpq_cfg,
            scale=scale,
            stats=guess_stats,
        )
        for key, value in guess_stats.items():
            stats[key] = stats.get(key, 0) + int(value)

        if t_guess is None:
            stats["t_stpq_miss"] = stats.get("t_stpq_miss", 0) + 1
            pairs.append((float(r_detection), np.nan))
            continue

        stats["t_detected"] += 1
        if source:
            source_key = f"source_{source}"
            stats[source_key] = stats.get(source_key, 0) + 1
        pairs.append((float(r_detection), float(t_guess)))

    return np.asarray(pairs, dtype=float), stats


def merge_stpq_t(
    features: pd.DataFrame,
    pairs: np.ndarray,
    *,
    max_r_distance_samples: int = 40,
) -> pd.DataFrame:
    """Overwrite only ``T_global_center_idx`` using nearest matched R centers."""

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
    stpq_r = pairs[:, 0]
    stpq_t = pairs[:, 1]
    assigned = np.full(len(output), np.nan, dtype=float)
    sources = np.full(len(output), "missing", dtype=object)
    used: set[int] = set()

    for cycle_idx, r_value in enumerate(detected_r):
        if not np.isfinite(r_value):
            continue
        distances = np.abs(stpq_r - r_value)
        for pair_idx in np.argsort(distances):
            pair_idx = int(pair_idx)
            if pair_idx in used:
                continue
            if distances[pair_idx] > max_r_distance_samples:
                break
            used.add(pair_idx)
            if np.isfinite(stpq_t[pair_idx]):
                assigned[cycle_idx] = stpq_t[pair_idx]
                sources[cycle_idx] = "record_stpq_hybrid"
            else:
                sources[cycle_idx] = "record_stpq_miss"
            break

    output["T_global_center_idx"] = assigned
    output["t_source"] = sources
    return output


class PyHEARTS:
    """Self-contained implementation of the validated T-only hybrid.

    ``species=None`` preserves the species-agnostic 2025 defaults used for the
    full SPH validation. ``species="human"`` uses the 2025 human preset used
    for the QTDB manual-annotation benchmark. Mouse processing keeps the 2025
    core and does not apply the human STPQ post-pass.
    """

    def __init__(
        self,
        sampling_rate: float,
        verbose: bool = False,
        plot: bool = False,
        cfg: ProcessCycleConfig | None = None,
        *,
        species: str | None = None,
        core_cfg: CoreProcessCycleConfig | None = None,
        apply_stpq_t: bool | None = None,
        **core_overrides: Any,
    ):
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if species not in (None, "human", "mouse"):
            raise ValueError("species must be None, 'human', or 'mouse'")

        self.sampling_rate = float(sampling_rate)
        self.verbose = bool(verbose)
        self.plot = bool(plot)
        self.species = species

        if core_cfg is None:
            if species == "human":
                core_cfg = CoreProcessCycleConfig.for_human()
            elif species == "mouse":
                core_cfg = CoreProcessCycleConfig.for_mouse()
            else:
                core_cfg = CoreProcessCycleConfig()

        self.core_cfg = core_cfg
        self.stpq_cfg = replace(
            cfg or ProcessCycleConfig.for_human_unified(),
            version=HYBRID_CONFIG_VERSION,
        )
        self.cfg = self.stpq_cfg
        self.apply_stpq_t = species != "mouse" if apply_stpq_t is None else bool(apply_stpq_t)
        self._core = CorePyHEARTS(
            sampling_rate=self.sampling_rate,
            verbose=self.verbose,
            plot=self.plot,
            cfg=self.core_cfg,
            **core_overrides,
        )
        self.last_stpq_stats: dict[str, Any] = {}
        self.output_df = pd.DataFrame()
        self.epochs_df = pd.DataFrame()

    def preprocess_signal(self, ecg_signal: np.ndarray, **kwargs: Any):
        """Run the validated 2025 preprocessing implementation."""

        return self._core.preprocess_signal(ecg_signal, **kwargs)

    def analyze_ecg(
        self,
        ecg_signal: np.ndarray,
        verbose: bool | None = None,
        plot: bool | None = None,
        **_: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run 2025 morphology fitting followed by the T-only STPQ post-pass."""

        features, epochs = self._core.analyze_ecg(
            np.asarray(ecg_signal, dtype=float),
            verbose=verbose,
            plot=plot,
        )
        output = features.copy()

        if self.apply_stpq_t and not output.empty:
            r_values = pd.to_numeric(output.get("R_global_center_idx"), errors="coerce").to_numpy(
                dtype=float
            )
            pairs, self.last_stpq_stats = detect_record_stpq_t(
                np.asarray(ecg_signal, dtype=float),
                r_values,
                self.sampling_rate,
                self.stpq_cfg,
            )
            output = merge_stpq_t(output, pairs)

        self.output_df = output
        self.epochs_df = epochs
        self._core.output_df = output
        self._core.output_dict = output.to_dict(orient="list")
        return output, epochs

    def compute_hrv_metrics(self):
        result = self._core.compute_hrv_metrics()
        self.hrv_metrics = self._core.hrv_metrics
        return result

    def save_output(self, file_id: str, results_dir: str):
        output_dir = Path(results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{file_id}_pyhearts.csv"
        self.output_df.to_csv(output_path, index=True, na_rep="NaN")
        metadata = {
            "pyhearts_version": HYBRID_CONFIG_VERSION,
            "pipeline": {
                "core": "pyhearts-2025-symmetric-gaussian",
                "t_detector": "human-unified-record-stpq",
                "t_output": "T_global_center_idx",
            },
            "sampling_rate_hz": self.sampling_rate,
            "species": self.species,
            "apply_stpq_t": self.apply_stpq_t,
            "core_config": asdict(self.core_cfg),
            "stpq_config": asdict(self.stpq_cfg),
            "last_stpq_stats": self.last_stpq_stats,
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

    def save_hrv_metrics(self, file_id: str, results_dir: str):
        return self._core.save_hrv_metrics(file_id, results_dir)

    def save_rr_intervals(self, file_id: str, results_dir: str):
        return self._core.save_rr_intervals(file_id, results_dir)

    def __getattr__(self, name: str):
        core = self.__dict__.get("_core")
        if core is not None:
            return getattr(core, name)
        raise AttributeError(name)


__all__ = [
    "HYBRID_CONFIG_VERSION",
    "PyHEARTS",
    "detect_record_stpq_t",
    "merge_stpq_t",
]
