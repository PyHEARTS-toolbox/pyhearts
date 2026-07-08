"""Tests for inverted_t negative_peak → DZC optional rescue."""

from __future__ import annotations

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_delineation import MedianBeatTemplate
from pyhearts.processing.t_inverted_dzc_rescue import (
    stt_voltage_percentile,
    try_inverted_dzc_rescue,
)


def test_stt_voltage_percentile_extremes():
    ecg = np.linspace(0.0, 1.0, 100)
    assert stt_voltage_percentile(ecg, 99, 0, 100) > 95
    assert stt_voltage_percentile(ecg, 0, 0, 100) < 5


def test_config_flag_off_by_default():
    cfg = ProcessCycleConfig.for_human_unified_template_prior_ensemble()
    assert cfg.record_inverted_dzc_rescue is False
    cfg2 = ProcessCycleConfig.for_human_unified_template_prior_ensemble_inverted_dzc()
    assert cfg2.record_inverted_dzc_rescue is True
    assert cfg2.record_inverted_dzc_rescue_volt_percentile_min == 95.0


def test_rescue_skips_non_inverted():
    fs = 250.0
    n = 500
    ecg = np.zeros(n)
    # inverted-looking dip after R
    r_idx, s_i, q_next = 100, 120, 280
    t_neg = 200
    ecg[t_neg] = -0.5
    tmpl = MedianBeatTemplate(
        template=np.zeros(100),
        pre_r_samples=20,
        r_center_idx=0,
        p_offset_samples=None,
        t_offset_samples=40.0,
        p_polarity="positive",
        t_polarity="negative",
        median_rr_samples=200,
        n_beats=5,
        valid=True,
        template_anchor="s_to_q",
        t_landmark_idx=40.0,
        p_landmark_idx=None,
        t_morphology="normal",
    )
    cfg = ProcessCycleConfig.for_human_unified_template_prior_ensemble_inverted_dzc()
    new_t, dec = try_inverted_dzc_rescue(
        ecg=ecg,
        r_idx=r_idx,
        s_i=s_i,
        q_next=q_next,
        fs=fs,
        current_t=float(t_neg),
        current_source="negative_peak",
        tmpl=tmpl,
        cfg=cfg,
    )
    assert new_t is None
    assert dec.reason == "morphology_not_inverted"


def test_match_ms_config_still_present():
    cfg = ProcessCycleConfig.for_human_unified_template_prior_ensemble_inverted_dzc()
    assert getattr(cfg, "record_inverted_dzc_rescue_match_ms", None) == 12.0
