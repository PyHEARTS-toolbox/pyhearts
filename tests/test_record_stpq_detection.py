"""Tests for record-level STPQ P/T beat detection."""

import numpy as np
import pytest
from dataclasses import replace

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_stpq_detection import (
    _apex_with_threshold,
    _apply_w1_hi_sq_frac_floor,
    _early_peak_stpq_tuning_applies,
    _early_peak_landmark_frac,
    _suppress_early_peak_beat_tuning,
    _inverted_plateau_apex_forward,
    _resolve_stpq_t_tpl_idx_for_projection,
    _search_p_template_guided,
    _search_t_early_peak_apex,
    _stpq_p_window_samples,
    _stpq_t_search_end_tpl_idx,
    _stpq_t_window_samples,
    _t_search_prefer_negative,
    defer_stpq_t_overwrite,
    p_pr_window_samples,
    project_t_center_sample,
    record_detect_t_peak,
    record_fallback_p_search,
    record_stpq_pt_guesses,
    stpq_t_use_biphasic_fallback,
    stpq_p_window_samples,
)
from pyhearts.processing.record_delineation import (
    build_stpq_beat_template,
    delineate_record_template,
)


def _synthetic_stpq_record(fs: float = 250.0, n_beats: int = 15):
    rr = int(0.8 * fs)
    length = rr * (n_beats + 2)
    sig = np.zeros(length, dtype=float)
    r_peaks = []
    for i in range(1, n_beats + 1):
        r = i * rr
        r_peaks.append(r)
        sig[r] += 1.2
        sig[r + int(0.04 * fs)] -= 0.35
        t0 = r + int(0.28 * fs)
        for k in range(int(0.02 * fs) + 1):
            sig[t0 + k] -= 0.25
        sig[r + int(-0.14 * fs)] += 0.12
    return sig, np.asarray(r_peaks, dtype=int)


class TestStpqWindows:
    def test_t_window_ordered(self):
        t_lo, t_hi = _stpq_t_window_samples(100, 300, 20.0, 180.0, 200)
        assert t_lo < t_hi
        assert 100 <= t_lo < 300

    def test_t_window_mode1_narrower_than_mid_tp(self):
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_t_w1_end_mode="template_tj_margin",
            record_stpq_t_w1_post_tj_frac=0.08,
        )
        t_lo_legacy, t_hi_legacy = _stpq_t_window_samples(
            100, 300, 20.0, 180.0, 200, None
        )
        t_lo, t_hi = _stpq_t_window_samples(100, 300, 20.0, 180.0, 200, cfg)
        assert t_lo == t_lo_legacy
        assert t_hi < t_hi_legacy

    def test_w1_hi_floor_extends_narrow_formula(self):
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_t_w1_end_mode="template_tj_margin",
            record_stpq_t_w1_post_tj_frac=0.15,
            record_stpq_w1_hi_min_sq_frac=0.40,
            record_stpq_w1_hi_pj_margin_sq_frac=0.15,
        )
        t_j, p_j, n_tpl = 16.0, 128.0, 200
        formula = _stpq_t_search_end_tpl_idx(t_j, p_j, cfg)
        assert formula / (n_tpl - 1) < 0.40
        floored = _apply_w1_hi_sq_frac_floor(formula, t_j, p_j, n_tpl, cfg)
        assert abs(floored / (n_tpl - 1) - 0.40) < 1e-9
        t_lo, t_hi = _stpq_t_window_samples(100, 300, t_j, p_j, n_tpl, cfg)
        assert t_hi > _stpq_t_window_samples(100, 300, t_j, p_j, n_tpl, replace(cfg, record_stpq_w1_hi_min_sq_frac=0.0))[1]

    def test_w1_hi_floor_leaves_wide_formula_unchanged(self):
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_t_w1_end_mode="mid_tp",
            record_stpq_w1_hi_min_sq_frac=0.40,
        )
        t_j, p_j, n_tpl = 20.0, 180.0, 200
        end = _stpq_t_search_end_tpl_idx(t_j, p_j, cfg)
        assert end / (n_tpl - 1) >= 0.40
        assert _apply_w1_hi_sq_frac_floor(end, t_j, p_j, n_tpl, cfg) == end

    def test_w1_hi_floor_respects_p_j_margin_cap(self):
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_w1_hi_min_sq_frac=0.40,
            record_stpq_w1_hi_pj_margin_sq_frac=0.15,
        )
        t_j, p_j, n_tpl = 16.0, 100.0, 200
        formula = 26.0
        floored = _apply_w1_hi_sq_frac_floor(formula, t_j, p_j, n_tpl, cfg)
        cap_frac = p_j / (n_tpl - 1) - 0.15
        assert floored / (n_tpl - 1) <= cap_frac + 1e-9

    def test_p_window_before_q(self):
        p_lo, p_hi = _stpq_p_window_samples(100, 300, 40.0, 160.0, 200)
        assert p_lo < p_hi <= 300

    def test_template_guided_picks_near_projected_center(self):
        fs = 250.0
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_p_r_anchor=True,
            record_stpq_p_r_anchor_mode="current_r",
            record_stpq_p_pr_min_ms=80.0,
            record_stpq_p_pr_max_ms=250.0,
            record_stpq_p_template_guided=True,
            record_stpq_use_savgol=False,
            p_t_search_savgol=False,
        )
        r_det = 300
        pr_ms = 180.0
        p_center = r_det - int(round(pr_ms * fs / 1000.0))
        n = 400
        ecg = np.zeros(n, dtype=float)
        # Late larger bump (wrong) and earlier smaller true P near template center
        ecg[p_center - int(0.08 * fs)] = 0.08
        true_p = p_center - int(0.02 * fs)
        ecg[true_p] = 0.15
        p_lo, p_hi = p_pr_window_samples(r_det, fs, cfg, signal_len=n)

        class _Tmpl:
            t_landmark_idx = 20
            p_landmark_idx = 90
            p_polarity = "positive"
            th_p_up = 0.2
            th_p_down = -0.2
            template = np.zeros(200)

        idx, _ = _search_p_template_guided(
            ecg, p_lo, p_hi, p_center, _Tmpl(), fs, cfg
        )
        assert idx is not None
        assert abs(idx - true_p) <= 3

    def test_template_guided_ignores_negative_template_polarity(self):
        fs = 250.0
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_p_r_anchor=True,
            record_stpq_p_r_anchor_mode="current_r",
            record_stpq_p_pr_min_ms=80.0,
            record_stpq_p_pr_max_ms=250.0,
            record_stpq_p_template_guided=True,
            record_stpq_use_savgol=False,
            p_t_search_savgol=False,
        )
        r_det = 500
        p_center = r_det - int(round(180.0 * fs / 1000.0))
        n = 700
        ecg = np.zeros(n, dtype=float)
        true_p = p_center - 2
        ecg[true_p] = 0.12
        # Stronger negative deflection closer to R (would win if polarity=min)
        ecg[p_center + int(0.06 * fs)] = -0.25
        p_lo, p_hi = p_pr_window_samples(r_det, fs, cfg, signal_len=n)

        class _Tmpl:
            p_polarity = "negative"
            template = np.zeros(200)

        idx, pol = _search_p_template_guided(
            ecg, p_lo, p_hi, p_center, _Tmpl(), fs, cfg
        )
        assert idx is not None
        assert abs(idx - true_p) <= 3
        assert pol == "positive"

    def test_p_window_current_r_covers_pr_interval(self):
        fs = 250.0
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_p_r_anchor=True,
            record_stpq_p_r_anchor_mode="current_r",
            record_stpq_p_pr_min_ms=80.0,
            record_stpq_p_pr_max_ms=250.0,
        )
        r_det = 1000
        p_lo, p_hi = stpq_p_window_samples(
            s_i=700,
            q_next=995,
            t_j=40.0,
            p_j=160.0,
            n_tpl=200,
            r_det=r_det,
            r_next=1200,
            sampling_rate=fs,
            cfg=cfg,
        )
        assert p_lo == r_det - int(round(250.0 * fs / 1000.0))
        assert p_hi == r_det - int(round(80.0 * fs / 1000.0))
        p_sample = r_det - int(round(140.0 * fs / 1000.0))
        assert p_lo <= p_sample <= p_hi

    def test_p_window_next_r_differs_from_current_r(self):
        fs = 250.0
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_p_r_anchor=True,
            record_stpq_p_r_anchor_mode="next_r",
            record_stpq_p_pr_min_ms=80.0,
            record_stpq_p_pr_max_ms=250.0,
        )
        r_det = 800
        r_next = 1000
        cur_lo, cur_hi = stpq_p_window_samples(
            s_i=700, q_next=995, t_j=40.0, p_j=160.0, n_tpl=200,
            r_det=r_det, r_next=r_next, sampling_rate=fs, cfg=replace(cfg, record_stpq_p_r_anchor_mode="current_r"),
        )
        nxt_lo, nxt_hi = stpq_p_window_samples(
            s_i=700, q_next=995, t_j=40.0, p_j=160.0, n_tpl=200,
            r_det=r_det, r_next=r_next, sampling_rate=fs, cfg=cfg,
        )
        assert (cur_lo, cur_hi) != (nxt_lo, nxt_hi)
        assert nxt_hi == r_next - int(round(80.0 * fs / 1000.0))


class TestRecordStpqDetection:
    def test_mean_template_option(self):
        fs = 250.0
        sig, r_peaks = _synthetic_stpq_record(fs=fs, n_beats=12)
        cfg_median = replace(
            ProcessCycleConfig(),
            record_template_anchor="s_to_q",
            record_template_aggregate="median",
            record_delineation_min_beats=5,
            delineation_baseline_method="median_record",
        )
        cfg_mean = replace(cfg_median, record_template_aggregate="mean")
        t_med = build_stpq_beat_template(sig, r_peaks, fs, cfg_median)
        t_mean = build_stpq_beat_template(sig, r_peaks, fs, cfg_mean)
        assert t_med.valid and t_mean.valid
        assert t_med.template.shape == t_mean.template.shape
        assert cfg_mean.record_template_aggregate == "mean"

    def test_record_stpq_pt_guesses_on_synthetic(self):
        fs = 250.0
        sig, r_peaks = _synthetic_stpq_record(fs=fs, n_beats=12)
        proc = replace(
            ProcessCycleConfig.for_human_unified(),
            record_delineation_min_beats=5,
            record_stpq_p_r_anchor=True,
            record_stpq_p_r_anchor_mode="current_r",
        )
        raw = build_stpq_beat_template(sig, r_peaks, fs, proc)
        tmpl = delineate_record_template(raw, fs, proc)
        assert tmpl.valid

        from pyhearts.processing.delineation_signal import prepare_record_delineation_signal

        ecg_delim = prepare_record_delineation_signal(sig, fs, proc)
        r_det = int(r_peaks[3])
        r_next = int(r_peaks[4])
        t_g, p_g = record_stpq_pt_guesses(
            ecg_delim, r_det, r_next, tmpl, fs, proc, scale=1.0
        )
        assert t_g is not None and p_g is not None
        assert r_det < t_g < r_next
        assert p_g < r_det
        pr_ms = (r_det - p_g) / fs * 1000.0
        assert 80.0 <= pr_ms <= 250.0

    def test_fallback_p_search_uses_pr_window(self):
        fs = 250.0
        sig, r_peaks = _synthetic_stpq_record(fs=fs, n_beats=12)
        cfg = replace(
            ProcessCycleConfig(),
            record_template_anchor="s_to_q",
            record_delineation_min_beats=5,
            delineation_baseline_method="median_record",
            record_stpq_p_r_anchor=True,
            record_stpq_p_r_anchor_mode="current_r",
            record_stpq_use_savgol=False,
            p_t_search_savgol=False,
        )
        raw = build_stpq_beat_template(sig, r_peaks, fs, cfg)
        tmpl = delineate_record_template(raw, fs, cfg)
        from pyhearts.processing.delineation_signal import prepare_record_delineation_signal

        ecg_delim = prepare_record_delineation_signal(sig, fs, cfg)
        r_det = int(r_peaks[5])
        p_fb = record_fallback_p_search(ecg_delim, r_det, int(r_peaks[6]), tmpl, fs, cfg)
        assert p_fb is not None
        pr_ms = (r_det - p_fb) / fs * 1000.0
        assert 80.0 <= pr_ms <= 250.0

    def test_unified_preset_enables_stpq_search(self):
        cfg = ProcessCycleConfig.for_human_unified()
        assert cfg.record_delineation_stpq_search is True
        assert cfg.record_template_anchor == "s_to_q"

    def test_t_rt_cap_rejects_late_placement(self):
        fs = 250.0
        n = 800
        ecg = np.zeros(n, dtype=float)
        r_det = 100
        s_i = 120
        q_next = 500
        ecg[r_det + int(0.60 * fs)] = 0.5
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_t_max_rt_ms=550.0,
            record_stpq_use_savgol=False,
            p_t_search_savgol=False,
        )

        class _Tmpl:
            t_landmark_idx = 10
            p_landmark_idx = 180
            th_t_up = 0.01
            th_t_down = -0.01
            t_morphology = "upright_t"
            template = np.zeros(200)

        t_idx, _ = record_detect_t_peak(
            ecg, s_i, q_next, _Tmpl(), fs, cfg, r_idx=r_det
        )
        assert t_idx is None or t_idx <= r_det + int(round(550.0 * fs / 1000.0))

    def test_inverted_plateau_apex_forward_argmin(self):
        fs = 250.0
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_t_plateau_apex_forward_ms=40.0,
        )
        s_i, q_next = 100, 400
        n_tpl = 200
        t_j_tpl = 40
        anchor = s_i + int(round(t_j_tpl / (n_tpl - 1) * (q_next - s_i)))
        ecg = np.zeros(500, dtype=float)
        trough = anchor + int(0.02 * fs)
        ecg[trough] = -0.4
        ecg[anchor] = -0.05
        # flat inverted plateau: same depth ahead of onset
        for k in range(1, int(0.012 * fs)):
            ecg[trough + k] = -0.4

        class _Tmpl:
            t_landmark_idx = t_j_tpl
            p_landmark_idx = 160
            t_landmark_source = "plateau_apex"
            t_morphology = "inverted_t"
            template = np.zeros(n_tpl)

        t_lo, t_hi = anchor - 5, anchor + int(0.08 * fs)
        out = _inverted_plateau_apex_forward(
            ecg, s_i, q_next, _Tmpl(), t_lo, t_hi, fs, cfg
        )
        assert out is not None
        idx, pol = out
        assert idx == trough + int(0.012 * fs) - 1
        assert pol == "negative"


class TestEarlyPeakLandmarkStpq:
    def test_delineated_projection_uses_landmark_not_late_offset(self):
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_t_project_from="delineated",
        )

        class _Tmpl:
            t_landmark_idx = 30.0
            t_offset_samples = 80.0
            t_landmark_source = "early_peak"

        assert _resolve_stpq_t_tpl_idx_for_projection(_Tmpl(), cfg) == 30.0

    def test_early_peak_window_narrower_than_w1_hi_floor(self):
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_t_w1_end_mode="template_tj_margin",
            record_stpq_t_w1_post_tj_frac=0.15,
            record_stpq_w1_hi_min_sq_frac=0.40,
        )

        class _Tmpl:
            t_landmark_source = "early_peak"
            t_morphology = "normal"
            t_polarity = "positive"
            template = np.zeros(200)

        t_j, p_j, n_tpl = 16.0, 128.0, 200
        from pyhearts.processing.record_stpq_detection import _stpq_t_search_end_tpl_idx

        end_early = _stpq_t_search_end_tpl_idx(t_j, p_j, cfg, early_peak=True)
        end_default = _stpq_t_search_end_tpl_idx(t_j, p_j, cfg, early_peak=False)
        assert end_early < end_default
        lo_early, hi_early = _stpq_t_window_samples(
            100, 300, t_j, p_j, n_tpl, cfg, tmpl=_Tmpl()
        )
        assert hi_early - lo_early >= max(3, int(round(0.06 * 200)))

    def test_early_peak_w1_min_span_when_narrow(self):
        cfg = replace(
            ProcessCycleConfig.for_human_unified(),
            record_stpq_t_w1_end_mode="template_tj_margin",
            record_stpq_t_early_peak_w1_post_tj_frac=0.05,
        )

        class _Tmpl:
            t_landmark_source = "early_peak"
            t_morphology = "normal"
            t_polarity = "positive"
            template = np.zeros(200)

        s_i, q_next = 100, 130
        t_j, p_j, n_tpl = 40.0, 160.0, 200
        lo, hi = _stpq_t_window_samples(s_i, q_next, t_j, p_j, n_tpl, cfg, tmpl=_Tmpl())
        min_span = max(3, int(round(0.06 * (q_next - s_i))))
        assert hi - lo >= min_span

    def test_early_peak_prefers_earlier_extremum_near_center(self):
        fs = 250.0
        cfg = replace(
            ProcessCycleConfig(),
            record_stpq_use_savgol=False,
            record_stpq_t_mode1_max_dist_ms=40.0,
            record_stpq_t_mode1_min_amp_frac=0.15,
            record_stpq_t_template_guided_half_window_ms=60.0,
        )
        s_i, q_next = 100, 400
        n_tpl = 200
        t_j_tpl = 40
        center = s_i + int(round(t_j_tpl / (n_tpl - 1) * (q_next - s_i)))
        ecg = np.zeros(500, dtype=float)
        early = center + int(0.01 * fs)
        late = center + int(0.06 * fs)
        ecg[early] = 0.25
        ecg[late] = 0.45

        class _Tmpl:
            t_landmark_idx = t_j_tpl
            p_landmark_idx = 160
            t_landmark_source = "early_peak"
            t_morphology = "normal"
            template = np.zeros(n_tpl)

        idx, _ = _search_t_early_peak_apex(
            ecg, s_i, q_next, center, _Tmpl(), fs, cfg
        )
        assert idx == early

    def test_record_detect_early_peak_not_late_lobe(self):
        fs = 250.0
        cfg = replace(
            ProcessCycleConfig.for_human_unified(),
            record_stpq_use_savgol=False,
            p_t_search_savgol=False,
        )
        s_i, q_next = 100, 400
        n_tpl = 200
        t_j_tpl = 45
        center = s_i + int(round(t_j_tpl / (n_tpl - 1) * (q_next - s_i)))
        ecg = np.zeros(500, dtype=float)
        early = center + int(0.01 * fs)
        late = center + int(0.05 * fs)
        ecg[early] = 0.30
        ecg[late] = 0.50

        class _Tmpl:
            t_landmark_idx = t_j_tpl
            p_landmark_idx = 160
            t_landmark_source = "early_peak"
            t_morphology = "normal"
            th_t_up = 0.05
            th_t_down = -0.05
            template = np.zeros(n_tpl)

        idx, _ = record_detect_t_peak(ecg, s_i, q_next, _Tmpl(), fs, cfg)
        assert idx == early

    def test_inverted_early_peak_skips_tuning_uses_threshold(self):
        fs = 250.0
        cfg = replace(
            ProcessCycleConfig.for_human_unified(),
            record_stpq_use_savgol=False,
            p_t_search_savgol=False,
        )
        s_i, q_next = 100, 400
        n_tpl = 200
        t_j_tpl = 40
        center = s_i + int(round(t_j_tpl / (n_tpl - 1) * (q_next - s_i)))
        ecg = np.zeros(500, dtype=float)
        early_neg = center + int(0.01 * fs)
        late_neg = center + int(0.08 * fs)
        ecg[early_neg] = -0.30
        ecg[late_neg] = -0.55

        class _Tmpl:
            t_landmark_idx = t_j_tpl
            p_landmark_idx = 160
            t_landmark_source = "early_peak"
            t_morphology = "inverted_t"
            t_polarity = "negative"
            th_t_up = 0.05
            th_t_down = -0.05
            template = np.zeros(n_tpl)

        assert not _early_peak_stpq_tuning_applies(_Tmpl())
        idx, pol = record_detect_t_peak(ecg, s_i, q_next, _Tmpl(), fs, cfg)
        assert idx == early_neg
        assert pol == "negative"


class TestNegativePolarityStpq:
    def test_suppress_beat_tuning_sel301_landmark_frac(self):
        class _Sel301:
            t_landmark_source = "early_peak"
            t_morphology = "normal"
            t_landmark_idx = 31
            template = np.zeros(149)

        class _Sel230:
            t_landmark_source = "early_peak"
            t_morphology = "normal"
            t_landmark_idx = 62
            template = np.zeros(184)

        assert _early_peak_landmark_frac(_Sel301()) <= 0.21
        assert _suppress_early_peak_beat_tuning(_Sel301())
        assert not _early_peak_stpq_tuning_applies(_Sel301())
        assert _early_peak_landmark_frac(_Sel230()) > 0.21
        assert not _suppress_early_peak_beat_tuning(_Sel230())

    def test_defer_sel114_class_only(self):
        class _Sel114:
            t_landmark_source = "early_peak"
            t_polarity = "negative"
            t_morphology = "normal"

        class _InvertedEarlyPeak:
            t_landmark_source = "early_peak"
            t_polarity = "negative"
            t_morphology = "inverted_t"

        assert defer_stpq_t_overwrite(_Sel114(), manual_ann_ext="q2c")
        assert not defer_stpq_t_overwrite(_Sel114(), manual_ann_ext="q1c")
        assert not defer_stpq_t_overwrite(_InvertedEarlyPeak(), manual_ann_ext="q2c")
        assert not stpq_t_use_biphasic_fallback(_Sel114())
        assert _t_search_prefer_negative(_Sel114())

    def test_negative_skips_w1_hi_floor(self):
        fs = 250.0
        cfg = replace(
            ProcessCycleConfig.for_human_unified(),
            record_stpq_w1_hi_min_sq_frac=0.40,
        )
        n_tpl = 200
        t_j, p_j = 40, 160

        class _NegTmpl:
            t_landmark_idx = t_j
            p_landmark_idx = p_j
            t_landmark_source = "normal"
            t_polarity = "negative"
            t_morphology = "normal"
            template = np.zeros(n_tpl)

        s_i, q_next = 100, 400
        w_neg = _stpq_t_window_samples(
            s_i, q_next, float(t_j), float(p_j), n_tpl, cfg, tmpl=_NegTmpl()
        )
        end_narrow = _stpq_t_search_end_tpl_idx(t_j, p_j, cfg, early_peak=False)
        assert w_neg[1] == s_i + int(round(end_narrow / (n_tpl - 1) * (q_next - s_i)))

    def test_negative_threshold_no_biphasic_positive(self):
        fs = 250.0
        cfg = replace(
            ProcessCycleConfig.for_human_unified(),
            record_stpq_use_savgol=False,
            p_t_search_savgol=False,
        )
        s_i, q_next = 100, 400
        n_tpl = 200
        t_j_tpl = 40
        center = s_i + int(round(t_j_tpl / (n_tpl - 1) * (q_next - s_i)))
        ecg = np.zeros(500, dtype=float)
        early_neg = center + int(0.01 * fs)
        late_pos = center + int(0.06 * fs)
        ecg[early_neg] = -0.25
        ecg[late_pos] = 0.45

        class _Tmpl:
            t_landmark_idx = t_j_tpl
            p_landmark_idx = 160
            t_landmark_source = "normal"
            t_polarity = "negative"
            t_morphology = "normal"
            th_t_up = 0.05
            th_t_down = -0.05
            template = np.zeros(n_tpl)

        idx, pol = record_detect_t_peak(ecg, s_i, q_next, _Tmpl(), fs, cfg)
        assert idx == early_neg
        assert pol == "negative"


class TestApexSignedPolarity:
    def test_normal_prefers_positive_over_larger_negative(self):
        seg = np.array([0.0, -0.20, 0.12, 0.05])
        idx, pol = _apex_with_threshold(
            seg, prefer="max", threshold_up=0.05, threshold_down=-0.05, check_biphasic=True
        )
        assert pol == "positive"
        assert idx == 2
        assert seg[idx] == 0.12

    def test_inverted_prefers_negative_over_larger_positive(self):
        seg = np.array([0.0, -0.10, 0.22, 0.05])
        idx, pol = _apex_with_threshold(
            seg, prefer="min", threshold_up=0.05, threshold_down=-0.05, check_biphasic=True
        )
        assert pol == "negative"
        assert idx == 1
        assert seg[idx] == -0.10

    def test_empty_same_pol_falls_back_to_full_candidates(self):
        # prefer=min but only positive passes threshold — must not return null
        seg = np.array([0.0, 0.15, 0.05])
        idx, pol = _apex_with_threshold(
            seg, prefer="min", threshold_up=0.10, threshold_down=-0.10, check_biphasic=True
        )
        assert idx == 1
        assert pol == "positive"
        assert seg[idx] == 0.15

    def test_empty_same_pol_max_prefers_negative_fallback(self):
        seg = np.array([0.0, -0.15, -0.05])
        idx, pol = _apex_with_threshold(
            seg, prefer="max", threshold_up=0.10, threshold_down=-0.10, check_biphasic=True
        )
        assert idx == 1
        assert pol == "negative"
