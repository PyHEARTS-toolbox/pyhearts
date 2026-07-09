"""Fast smoke tests for CI (<30s total)."""

import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from pyhearts import PyHEARTS, ProcessCycleConfig


def _short_ecg(sampling_rate: float = 500.0, beats: int = 5) -> np.ndarray:
    """~3 s synthetic ECG with clear R-peaks."""
    rr = int(0.8 * sampling_rate)
    n = rr * (beats + 1)
    sig = np.zeros(n, dtype=float)
    for i in range(1, beats + 1):
        r = i * rr
        sig[r] = 1.0
        sig[r + int(0.04 * sampling_rate)] = -0.3
        sig[r + int(0.22 * sampling_rate)] = -0.2
        sig[r - int(0.12 * sampling_rate)] = 0.12
    return sig


def test_default_config_instantiates():
    ProcessCycleConfig()


def test_human_species_uses_unified_preset():
    cfg = PyHEARTS(sampling_rate=500.0, species="human").cfg
    assert cfg.version == "human-unified"
    assert cfg.record_delineation is True


@pytest.mark.smoke
def test_analyze_ecg_smoke_lite():
    """One end-to-end beat on a short trace (lite_mode for speed)."""
    fs = 500.0
    cfg = replace(
        ProcessCycleConfig.for_human_unified(),
        lite_mode=True,
        record_delineation_min_beats=3,
    )
    out, epochs = PyHEARTS(sampling_rate=fs, cfg=cfg).analyze_ecg(_short_ecg(fs))
    assert isinstance(out, pd.DataFrame)
    assert isinstance(epochs, pd.DataFrame)
    assert len(out) >= 1
