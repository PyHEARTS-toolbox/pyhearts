"""Tests for sel16420-style post_apex_dz vs positive_peak STPQ ablation."""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_delineation import MedianBeatTemplate
from pyhearts.processing.record_post_apex_dz_morphology import (
    classify_post_apex_dz_preference_template,
    downslope_dz_after_positive_peak,
    probe_post_apex_dz_segment_fraction,
    qualified_post_apex_dz_pair,
)
from pyhearts.processing.record_stpq_detection import (
    _search_t_post_apex_dz_vs_positive_peak,
    record_detect_t_peak,
)


def _mock_post_apex_tmpl(n_tpl: int = 200) -> MedianBeatTemplate:
    template = np.zeros(n_tpl)
    template[50:80] = np.linspace(0, 0.4, 30)
    template[80] = 0.5
    template[81:100] = np.linspace(0.45, 0.1, 19)
    return MedianBeatTemplate(
        template=template,
        pre_r_samples=50,
        r_center_idx=0,
        p_offset_samples=None,
        t_offset_samples=55.0,
        p_polarity="positive",
        t_polarity="positive",
        median_rr_samples=200,
        n_beats=5,
        valid=True,
        template_anchor="s_to_q",
        t_landmark_idx=58.0,
        p_landmark_idx=170.0,
        t_morphology="normal",
        t_landmark_source="early_peak",
        t_post_apex_dz_preference=True,
    )


def _plateau_wave(n: int = 120) -> np.ndarray:
    wave = np.zeros(n)
    wave[20:50] = np.linspace(0.0, 0.9, 30)
    wave[50:62] = 0.9
    wave[62:] = np.linspace(0.9, 0.1, n - 62)
    return wave


class TestPostApexDzPreference:
    def test_classify_requires_experiment_flag(self):
        tmpl = _mock_post_apex_tmpl()
        cfg_off = ProcessCycleConfig.for_human_unified()
        cfg_on = replace(
            ProcessCycleConfig.for_human_unified_post_apex_dz_preference(),
            record_stpq_post_apex_dz_max_beat_frac=None,
        )
        segments = [_plateau_wave()]
        assert not classify_post_apex_dz_preference_template(
            tmpl, cfg_off, 250.0, beat_segments=segments
        )
        assert classify_post_apex_dz_preference_template(
            tmpl, cfg_on, 250.0, beat_segments=segments
        )

    def test_downslope_dz_after_positive_peak(self):
        lo = 40
        wave = _plateau_wave(80)
        apex_abs = lo + 44
        dz = downslope_dz_after_positive_peak(wave, lo, after_abs=apex_abs)
        assert dz is not None
        assert int(dz) >= int(apex_abs)

    def test_qualified_pair_respects_late_window(self):
        lo = 0
        wave = _plateau_wave(120)
        pos_abs = 50
        dz_abs = 62
        cfg = ProcessCycleConfig.for_human_unified_post_apex_dz_preference()
        baseline = 0.0
        assert qualified_post_apex_dz_pair(
            wave, lo, pos_abs, dz_abs, baseline, 250.0, cfg
        )

    def test_probe_fraction_on_plateau_segment(self):
        cfg = replace(
            ProcessCycleConfig.for_human_unified_post_apex_dz_preference(),
            record_stpq_use_savgol=False,
        )
        frac = probe_post_apex_dz_segment_fraction([_plateau_wave()], 250.0, cfg)
        assert frac >= 0.99

    def test_compare_prefers_post_apex_dz(self, monkeypatch):
        fs = 250.0
        ecg = np.zeros(400)
        lo = 100
        ecg[lo : lo + 120] = _plateau_wave(120)
        apex_abs = lo + 50
        tmpl = _mock_post_apex_tmpl()
        cfg = replace(
            ProcessCycleConfig.for_human_unified_post_apex_dz_preference(),
            record_stpq_use_savgol=False,
            record_stpq_post_apex_dz_pos_early_ms=0.0,
        )

        def _fake_guided(*_args, **_kwargs):
            return apex_abs, "positive"

        monkeypatch.setattr(
            "pyhearts.processing.record_stpq_detection._search_t_template_guided",
            _fake_guided,
        )
        idx, pol = _search_t_post_apex_dz_vs_positive_peak(
            ecg, lo, lo + 119, apex_abs + 20, tmpl, fs, cfg
        )
        assert pol == "positive"
        assert idx is not None
        assert int(idx) > int(apex_abs)

    def test_record_detect_uses_branch_when_flagged(self):
        fs = 250.0
        ecg = np.zeros(600)
        s_i, q_next, r_idx = 120, 400, 100
        ecg[200:280] = np.linspace(0, 0.5, 80)
        ecg[280:320] = np.linspace(0.48, 0.1, 40)
        tmpl = _mock_post_apex_tmpl()
        cfg = ProcessCycleConfig.for_human_unified_post_apex_dz_preference()
        idx, _ = record_detect_t_peak(
            ecg, s_i, q_next, tmpl, fs, cfg, r_idx=r_idx
        )
        assert idx is not None
