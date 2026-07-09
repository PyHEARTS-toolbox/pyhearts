"""Tests for human_unified production preset."""

import pytest

from pyhearts.config import ProcessCycleConfig


def test_human_unified_production_defaults():
    cfg = ProcessCycleConfig.for_human_unified()
    assert cfg.version == "human-unified"
    assert cfg.record_delineation is True
    assert cfg.r_detection_method == "derivative"


def test_human_unified_requires_record_delineation_for_record_only():
    with pytest.raises(ValueError, match="record_only"):
        ProcessCycleConfig(
            record_delineation=False,
            p_t_detection_method="record_only",
        )
