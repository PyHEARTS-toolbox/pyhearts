"""Tests for WFDB lead selection (Sprint 0)."""

from __future__ import annotations

import pytest

from pyhearts.io.wfdb_lead import (
    pick_lead_index,
    pick_manual_annotation_ext,
)
from pyhearts.processing.fiducial_provenance import (
    init_fiducial_provenance,
    set_run_lead_metadata,
    set_wave_source,
)


@pytest.mark.parametrize(
    "sig_names,policy,expected_idx",
    [
        (["MLII", "V5"], "ecg2_else_ecg1", 0),
        (["MLII", "V5"], "first", 0),
        (["MLII", "V5"], "second", 1),
        (["MLII", "V5"], "limb_preferred", 0),
        (["ECG1", "ECG2"], "ecg2_else_ecg1", 1),
        (["ECG1", "ECG2"], "first", 0),
        (["V5"], "ecg2_else_ecg1", 0),
    ],
)
def test_pick_lead_index(sig_names, policy, expected_idx):
    assert pick_lead_index(sig_names, policy) == expected_idx


@pytest.mark.parametrize(
    "sig_names,policy,expected_ext",
    [
        (["MLII", "V5"], "ecg2_else_ecg1", "q1c"),
        (["ECG1", "ECG2"], "ecg2_else_ecg1", "q2c"),
        (["V5"], "ecg2_else_ecg1", "q1c"),
    ],
)
def test_pick_manual_annotation_ext(sig_names, policy, expected_ext):
    assert pick_manual_annotation_ext(sig_names, policy) == expected_ext


def test_pick_manual_annotation_ext_q2c_missing_falls_back_to_q1c():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    path = repo / "data" / "qtdb" / "1.0.0" / "sel16273"
    if not path.with_suffix(".hea").exists():
        pytest.skip("QTDB sel16273 not available")
    hdr_names = ["MLII", "V5"]  # policy picks ch1 -> q2c preferred
    assert pick_manual_annotation_ext(hdr_names, "second", record_path=path) == "q1c"


def test_fiducial_provenance_sources():
    out = {}
    init_fiducial_provenance(out, 2)
    set_wave_source(out, 0, "P", "per_cycle", confidence="high")
    set_wave_source(out, 1, "T", "record_stpq", confidence="high")
    set_run_lead_metadata(
        out, lead_index=0, lead_name="MLII", lead_policy="ecg2_else_ecg1"
    )
    assert out["p_source"][0] == "per_cycle"
    assert out["t_source"][1] == "record_stpq"
    assert out["wfdb_lead_index"] == [0, 0]
    assert out["wfdb_lead_name"] == ["MLII", "MLII"]
    assert out["lead_policy"] == ["ecg2_else_ecg1", "ecg2_else_ecg1"]
