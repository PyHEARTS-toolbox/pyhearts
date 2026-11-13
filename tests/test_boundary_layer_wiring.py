"""PyHEARTS.analyze_ecg takes clinical boundaries/intervals from the boundary layer.

Uses NeuroKit2's simulator (optional dependency) for a realistic 20 s ECG.
"""

import json

import numpy as np
import pytest

from pyhearts import PyHEARTS
from pyhearts.core.analyzer import BOUNDARY_COLUMNS, LAYER_INTERVALS, WAVES

nk = pytest.importorskip("neurokit2")


@pytest.fixture(scope="module")
def human_run(tmp_path_factory):
    fs = 500
    ecg = nk.ecg_simulate(duration=20, sampling_rate=fs, heart_rate=70, random_state=1)
    ph = PyHEARTS(sampling_rate=fs, species="human")
    x = ph.preprocess_signal(ecg, highpass_cutoff=0.5, lowpass_cutoff=50.0, filter_order=4,
                             notch_frequency=50.0, quality_factor=30.0)
    features, _ = ph.analyze_ecg(x)
    out = tmp_path_factory.mktemp("out")
    ph.save_output("sim", str(out))
    return ph, features, out


def test_gaussian_widths_renamed(human_run):
    _, f, _ = human_run
    for w in WAVES:
        assert f"{w}_width_ms" in f
        if w != "P":
            assert f"{w}_duration_ms" not in f


def test_boundary_and_interval_columns(human_run):
    _, f, _ = human_run
    for col in list(BOUNDARY_COLUMNS.values()) + list(LAYER_INTERVALS):
        assert col in f
    assert len(f) >= 15
    # Plausibility only. ECGSYN beats are not physiological templates (NeuroKit2's
    # own delineator measures ~140 ms QRS on this signal); accuracy is validated
    # against expert annotations in validation/ (LUDB, QTDB) and PTB-XL+.
    for col in LAYER_INTERVALS:
        assert f[col].notna().mean() >= 0.8, col
    assert 40 <= f["QRS_interval_ms"].median() <= 250
    assert f["PR_interval_ms"].median() > f["P_duration_ms"].median()
    assert f["QT_interval_ms"].median() > f["QRS_interval_ms"].median()
    # onset < offset ordering
    ok = f[["QRS_onset_global_idx", "QRS_offset_global_idx"]].dropna()
    assert (ok["QRS_onset_global_idx"] < ok["QRS_offset_global_idx"]).all()


def test_intervals_match_boundaries(human_run):
    ph, f, _ = human_run
    ms = 1000.0 / ph.sampling_rate
    d = (f["QRS_offset_global_idx"] - f["QRS_onset_global_idx"]) * ms
    both = d.notna() & f["QRS_interval_ms"].notna()
    np.testing.assert_allclose(d[both], f.loc[both, "QRS_interval_ms"])


def test_metadata_records_interval_source(human_run):
    _, _, out = human_run
    meta = json.loads((out / "sim_meta.json").read_text())
    assert meta["intervals"]["source"] == "boundary_layer"
    assert "boundary_params" in meta["intervals"]


def test_mouse_keeps_gaussian_intervals():
    ph = PyHEARTS(sampling_rate=10000, species="mouse")
    assert ph.interval_source == "gaussian_edges"
