"""Fast smoke tests for CI (<30s total)."""

import numpy as np
import pandas as pd
import pytest

from pyhearts import PyHEARTS
from pyhearts.config import ProcessCycleConfig


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


def test_human_species_uses_unified_pipeline():
    analyzer = PyHEARTS(sampling_rate=500.0, species="human")
    assert analyzer.pipeline_version == "morphology-record-t"
    assert analyzer.cfg.version == "v1-human"
    assert analyzer.apply_record_t is True


@pytest.mark.smoke
def test_analyze_ecg_smoke():
    """End-to-end analysis returns fitted cycles and record-level T centers."""
    fs = 500.0
    out, epochs = PyHEARTS(sampling_rate=fs, species="human").analyze_ecg(_short_ecg(fs, beats=8))
    assert isinstance(out, pd.DataFrame)
    assert isinstance(epochs, pd.DataFrame)
    assert len(out) >= 1
    assert "T_global_center_idx" in out
    assert "T_gaussian_global_center_idx" in out
    assert "t_source" in out
