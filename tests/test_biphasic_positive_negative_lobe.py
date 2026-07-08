"""Tests for biphasic +− template classify and positive-lobe T search."""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_delineation import (
    MedianBeatTemplate,
    build_stpq_beat_template,
    delineate_record_template,
)
from pyhearts.processing.record_stpq_detection import (
    _search_t_biphasic_positive_lobe,
    record_detect_t_peak,
)
from pyhearts.processing.record_template_biphasic import (
    MORPH_BIPHASIC_POS_NEG,
    classify_biphasic_positive_negative,
)


def _synthetic_biphasic_pm_record(fs: float = 250.0, n_beats: int = 12):
    rr = int(0.8 * fs)
    length = rr * (n_beats + 2)
    sig = np.zeros(length, dtype=float)
    r_peaks = []
    for i in range(1, n_beats + 1):
        r = i * rr
        r_peaks.append(r)
        sig[r] += 1.0
        sig[r + int(0.04 * fs)] -= 0.3
        # Early shallow positive then later negative (biphasic +- on template)
        t_pos = r + int(0.22 * fs)
        width = int(0.04 * fs)
        for k in range(width):
            sig[t_pos + k] += 0.35 * (1.0 - abs(k - width / 2) / (width / 2))
        t_neg = r + int(0.38 * fs)
        for k in range(width):
            sig[t_neg + k] -= 0.45 * (1.0 - abs(k - width / 2) / (width / 2))
    return sig, np.asarray(r_peaks, dtype=int)


class TestBiphasicClassification:
    def test_classify_positive_before_negative(self):
        cfg = replace(
            ProcessCycleConfig(),
            record_template_detect_biphasic_positive_negative=True,
            record_template_t_morphology_sq_frac=(0.15, 0.75),
        )
        n = 200
        tpl = np.zeros(n)
        tpl[50:70] += 0.5
        tpl[120:140] -= 0.6
        tag, pos, neg = classify_biphasic_positive_negative(tpl, cfg, 250.0)
        assert tag == MORPH_BIPHASIC_POS_NEG
        assert pos is not None and neg is not None
        assert pos < neg

    def test_disabled_leaves_unchanged(self):
        cfg = ProcessCycleConfig()
        tpl = np.zeros(100)
        tpl[30] = 1.0
        tpl[70] = -1.0
        tag, pos, neg = classify_biphasic_positive_negative(tpl, cfg, 250.0)
        assert tag == "unchanged"
        assert pos is None


class TestBuildStpqTemplate:
    def test_build_stpq_sets_biphasic_landmarks(self):
        sig, r_peaks = _synthetic_biphasic_pm_record()
        cfg = ProcessCycleConfig.for_human_unified_biphasic_positive_negative_lobe_search()
        tmpl = build_stpq_beat_template(sig, r_peaks, 250.0, cfg)
        assert tmpl.valid
        assert tmpl.t_morphology == MORPH_BIPHASIC_POS_NEG
        assert tmpl.t_landmark_source == "biphasic_positive_apex"
        assert tmpl.t_biphasic_pos_landmark_idx is not None
        assert tmpl.t_biphasic_neg_landmark_idx is not None
        assert tmpl.t_landmark_idx == tmpl.t_biphasic_pos_landmark_idx

    def test_baseline_does_not_set_biphasic(self):
        sig, r_peaks = _synthetic_biphasic_pm_record()
        cfg = ProcessCycleConfig.for_human_unified()
        tmpl = build_stpq_beat_template(sig, r_peaks, 250.0, cfg)
        assert tmpl.t_morphology != MORPH_BIPHASIC_POS_NEG

    def test_delineate_preserves_biphasic_landmarks(self):
        sig, r_peaks = _synthetic_biphasic_pm_record()
        cfg = ProcessCycleConfig.for_human_unified_biphasic_positive_negative_lobe_search()
        raw = build_stpq_beat_template(sig, r_peaks, 250.0, cfg)
        tmpl = delineate_record_template(raw, 250.0, cfg)
        assert tmpl.t_morphology == MORPH_BIPHASIC_POS_NEG
        assert tmpl.t_biphasic_pos_landmark_idx == raw.t_biphasic_pos_landmark_idx
        assert tmpl.t_biphasic_neg_landmark_idx == raw.t_biphasic_neg_landmark_idx


class TestPositiveLobeSearch:
    def _mock_tmpl(self, n_tpl: int = 200, pos_frac: float = 0.35, neg_frac: float = 0.65):
        return MedianBeatTemplate(
            template=np.zeros(n_tpl),
            pre_r_samples=50,
            r_center_idx=0,
            p_offset_samples=None,
            t_offset_samples=None,
            p_polarity="positive",
            t_polarity="positive",
            median_rr_samples=200,
            n_beats=10,
            valid=True,
            template_anchor="s_to_q",
            t_landmark_idx=pos_frac * (n_tpl - 1),
            p_landmark_idx=0.85 * (n_tpl - 1),
            t_morphology=MORPH_BIPHASIC_POS_NEG,
            t_landmark_source="biphasic_positive_apex",
            t_biphasic_pos_landmark_idx=pos_frac * (n_tpl - 1),
            t_biphasic_neg_landmark_idx=neg_frac * (n_tpl - 1),
        )

    def test_search_stays_before_negative_lobe(self):
        fs = 250.0
        length = 800
        ecg = np.zeros(length)
        s_i, q_next = 100, 350
        # Positive lobe ~200-260, negative ~320-380
        ecg[200:260] += 0.4
        ecg[320:380] -= 0.5
        cfg = ProcessCycleConfig.for_human_unified_biphasic_positive_negative_lobe_search()
        tmpl = self._mock_tmpl(pos_frac=0.40, neg_frac=0.75)
        idx, pol = _search_t_biphasic_positive_lobe(ecg, s_i, q_next, tmpl, fs, cfg)
        assert idx is not None
        assert pol == "positive"
        t_neg = int(s_i + 0.75 * (q_next - s_i))
        assert idx < t_neg - int(0.03 * fs)

    def test_record_detect_uses_lobe_search(self):
        fs = 250.0
        length = 800
        ecg = np.zeros(length)
        s_i, q_next = 100, 350
        ecg[210:250] += 0.5
        ecg[330:370] -= 0.6
        cfg = ProcessCycleConfig.for_human_unified_biphasic_positive_negative_lobe_search()
        assert cfg.record_biphasic_pm_lobe_search
        tmpl = self._mock_tmpl(pos_frac=0.38, neg_frac=0.72)
        idx, pol = record_detect_t_peak(
            ecg, s_i, q_next, tmpl, fs, cfg, r_idx=s_i + 50
        )
        assert idx is not None
        assert idx < s_i + int(0.55 * (q_next - s_i))
