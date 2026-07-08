"""Tests for record-only P/T detection mode."""

from dataclasses import replace

import numpy as np
import pytest

from pyhearts import PyHEARTS
from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.pt_detection_mode import p_t_detection_is_record_only


def _record_only_cfg(**kwargs) -> ProcessCycleConfig:
    return replace(
        ProcessCycleConfig.for_human_unified(),
        p_t_detection_method="record_only",
        **kwargs,
    )


class TestRecordOnlyPt:
    def test_record_only_requires_record_delineation(self):
        with pytest.raises(ValueError, match="record_only"):
            ProcessCycleConfig(
                p_t_detection_method="record_only",
                record_delineation=False,
            )

    def test_record_only_preset_fields(self):
        cfg = _record_only_cfg()
        assert cfg.p_t_detection_method == "record_only"
        assert cfg.record_delineation is True
        assert cfg.record_template_anchor == "s_to_q"
        assert cfg.record_delineation_stpq_search is True

    def test_analyze_ecg_record_only_smoke(self):
        fs = 250.0
        rr = int(0.8 * fs)
        sig = np.zeros(rr * 12, dtype=float)
        for i in range(1, 11):
            r = i * rr
            sig[r] = 1.0
            sig[r + int(0.04 * fs)] = -0.3
            sig[r + int(0.25 * fs)] = -0.2
            sig[r - int(0.12 * fs)] = 0.15
        cfg = _record_only_cfg(record_delineation_min_beats=3, lite_mode=True)
        assert p_t_detection_is_record_only(cfg)
        out, _ = PyHEARTS(sampling_rate=fs, cfg=cfg, verbose=False).analyze_ecg(sig)
        assert len(out) >= 1
