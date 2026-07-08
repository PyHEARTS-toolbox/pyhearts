"""Tests for biphasic +− classify-only early-T guardrail (no lobe search)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_delineation import (
    MedianBeatTemplate,
    build_stpq_beat_template,
)
from pyhearts.processing.record_template_biphasic import (
    MORPH_BIPHASIC_POS_NEG,
    apply_biphasic_pm_early_t_guardrail,
)
from tests.test_biphasic_positive_negative_lobe import _synthetic_biphasic_pm_record


def _mock_biphasic_tmpl(n_tpl: int = 200, pos_frac: float = 0.40) -> MedianBeatTemplate:
    pos = pos_frac * (n_tpl - 1)
    return MedianBeatTemplate(
        template=np.zeros(n_tpl),
        pre_r_samples=50,
        r_center_idx=0,
        p_offset_samples=None,
        t_offset_samples=80.0,
        p_polarity="positive",
        t_polarity="positive",
        median_rr_samples=200,
        n_beats=5,
        valid=True,
        template_anchor="s_to_q",
        t_landmark_idx=50.0,
        p_landmark_idx=0.85 * (n_tpl - 1),
        t_morphology=MORPH_BIPHASIC_POS_NEG,
        t_landmark_source="rising_edge",
        t_biphasic_pos_landmark_idx=pos,
        t_biphasic_neg_landmark_idx=pos + 20,
    )


class TestGuardrailPreset:
    def test_classify_without_lobe_override(self):
        sig, r_peaks = _synthetic_biphasic_pm_record()
        cfg = ProcessCycleConfig.for_human_unified_biphasic_pm_early_guardrail()
        tmpl = build_stpq_beat_template(sig, r_peaks, 250.0, cfg)
        assert tmpl.t_morphology == MORPH_BIPHASIC_POS_NEG
        assert tmpl.t_biphasic_pos_landmark_idx is not None
        assert not cfg.record_biphasic_pm_lobe_search
        assert tmpl.t_landmark_source != "biphasic_positive_apex"

    def test_guardrail_clamps_only_when_too_early(self, monkeypatch):
        fs = 250.0
        first_pos = 250
        margin_samp = int(round(10.0 * fs / 1000.0))

        monkeypatch.setattr(
            "pyhearts.processing.record_delineation._stpq_s_q_anchor_indices",
            lambda *a, **k: (100, 350),
        )
        monkeypatch.setattr(
            "pyhearts.processing.record_stpq_detection._tpl_index_to_sample",
            lambda s, q, tpl, n: first_pos,
        )
        monkeypatch.setattr(
            "pyhearts.processing.record_delineation._sync_peak",
            lambda output_dict, cycle_idx, wave, peak, one_cycle, sampling_rate, cfg: peak,
        )
        monkeypatch.setattr(
            "pyhearts.processing.record_fiducial_smoothing._sync_cycle_relative_peak",
            lambda *a, **k: None,
        )

        tmpl = _mock_biphasic_tmpl()
        cfg = ProcessCycleConfig.for_human_unified_biphasic_pm_early_guardrail()
        ecg = np.zeros(600)
        ecg[first_pos] = 0.6
        too_early = float(first_pos) - margin_samp - 5
        ok_late = float(first_pos) + 5.0
        way_too_early = float(first_pos) - 200.0
        output_dict = {
            "T_global_center_idx": [too_early, ok_late, way_too_early],
            "R_global_center_idx": [200.0, 500.0, 800.0],
        }
        cycles = np.array([0, 1, 2])
        epochs_df = pd.DataFrame(
            {"cycle": [0, 1, 2], "index": [0, 1, 2], "signal_y": [0.0, 0.0, 0.0]}
        )
        stats = apply_biphasic_pm_early_t_guardrail(
            output_dict,
            epochs_df,
            cycles,
            np.array([200, 500, 800]),
            ecg,
            tmpl,
            fs,
            cfg,
        )
        assert stats["adjusted"] == 1
        assert output_dict["T_global_center_idx"][0] == float(first_pos)
        assert output_dict["T_global_center_idx"][1] == ok_late
        assert output_dict["T_global_center_idx"][2] == way_too_early
